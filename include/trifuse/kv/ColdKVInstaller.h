#pragma once
#include "ColdKVEncoder.h"
#include "ThresholdPolicy.h"
#include <executorch/backends/mlx/runtime/MLXSequenceCache.h>
#include <mlx/memory.h>
#include <mlx/allocator.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <thread>

// The cache and all MLX operations remain owner-confined. Only independent,
// bounded FP16 snapshots enter the low-priority codec worker. No worker keeps
// a cache view alive (which would disable donation and copy the whole pool).
struct ColdKVInstaller {
  using Cache = executorch::backends::mlx::MLXSequenceCache;
  using ColdLayer = executorch::backends::mlx::ColdLayer;
  using Policy = trifuse_cold_kv_experiment::ThresholdPolicy;
  using Clock = std::chrono::steady_clock;
  struct Delta {
    std::vector<uint32_t> k, v;
    std::vector<uint16_t> ks, kb, vs, vb;
  };
  struct Job {
    uint64_t epoch;
    int from, to, next = 0, completed = 0, missed = 0, layer_count = 0;
    int candidate_layer = -1;
    size_t original_bytes = 0;
    double layer_publish_ms = 0;
    std::unique_ptr<ColdLayer> candidate;
    bool inflight = false;
    Clock::time_point began = Clock::now();
    double owner_ms = 0;
    std::vector<Delta> deltas;
    std::deque<std::shared_ptr<ColdKVEncoder::Snapshot>> snapshots;
    int per_layer() const { return (to - from) / 256; }
    int total() const { return layer_count * per_layer(); }
  };
  std::unique_ptr<ColdKVEncoder> encoder;
  std::unique_ptr<Policy> policy;
  std::unique_ptr<Job> job;
  uint64_t epoch = 1, snapshot_deferrals = 0, snapshot_drains = 0, publications = 0;
  uint64_t ticks_while_encoding = 0, discarded_jobs = 0, max_snapshot_bytes = 0, max_delta_bytes = 0;
  int layers = 0, recent = 512, cold_tokens = 0, blocks = 0;
  // Owner-confined scalar observations only: no extra tensor roots/evaluation.
  // These positions delimit a conservative post-maintenance decode suffix.
  int last_trigger_position = -1, last_completed_position = -1;
  uint64_t request_publications = 0;
  bool layered = true;
  uint64_t layer_publications = 0, max_candidate_bytes = 0, old_prefix_copy_bytes = 0;
  int snapshot_batch = 4, defer_ticks = 4;
  size_t before_bytes = 0, after_bytes = 0, active_before = 0, active_after = 0;
  double transaction_ms = 0, snapshot_ms = 0, snapshot_wait_ms = 0, publish_ms = 0;
  double encode_ms = 0, predict_ms = 0, pack_ms = 0, collect_ms = 0, reset_wait_ms = 0;
  static double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
  }
  static int option(const char* name, int fallback, int lo, int hi) {
    const char* value = std::getenv(name); if (!value) return fallback;
    size_t used = 0; const std::string text(value); const int n = std::stoi(text, &used);
    if (used != text.size() || n < lo || n > hi)
      throw std::invalid_argument(std::string(name) + " outside allowed range");
    return n;
  }
  size_t delta_bytes() const {
    size_t bytes=0; if (job) for (const auto& d:job->deltas)
      bytes+=sizeof(uint32_t)*(d.k.capacity()+d.v.capacity())+
          sizeof(uint16_t)*(d.ks.capacity()+d.kb.capacity()+d.vs.capacity()+d.vb.capacity());
    return bytes;
  }
  explicit ColdKVInstaller(int count) : layers(count) {
    const char* path = std::getenv("TRIFUSE_COLD_KV_MODEL");
    if (!path || !*path) return;
    layered = option("TRIFUSE_COLD_KV_LAYERED",1,0,1) != 0;
    recent = option("TRIFUSE_COLD_KV_RECENT",512,256,4096);
    snapshot_batch = option("TRIFUSE_COLD_KV_SNAPSHOT_BATCH",4,1,8);
    defer_ticks = option("TRIFUSE_COLD_KV_DEFER_TICKS",4,0,64);
    Policy::Config config;
    config.high_bytes = uint64_t(option("TRIFUSE_COLD_KV_THRESHOLD_MIB",448,256,4096))*Policy::kMiB;
    config.max_high_bytes = uint64_t(option("TRIFUSE_COLD_KV_MAX_THRESHOLD_MIB",512,256,4096))*Policy::kMiB;
    config.adaptive = option("TRIFUSE_COLD_KV_ADAPTIVE",1,0,1) != 0;
    policy = std::make_unique<Policy>(config, Policy::Geometry{count,16,128},recent,256);
    encoder = std::make_unique<ColdKVEncoder>(path);
  }
  // Runs at owner boundaries before a forward. Below threshold there is only
  // metadata accounting; no GPU sync, KV reads, or snapshots. A request ending
  // neither triggers compression nor waits for the codec. Already-ready work
  // may publish on close; unfinished work survives for the same cache owner.
  void tick_legacy(Cache& cache, const std::function<void()>& drain, bool supply = true) {
    namespace mx = mlx::core;
    if (!encoder) return;
    if (!job) {
      if (!supply) return;
      const auto decision = policy->evaluate(cache.allocated_payload_bytes(), cache.length(), cache.cold_prefix_length());
      if (!decision.trigger) return;
      last_trigger_position = cache.length();
      job = std::make_unique<Job>();
      job->epoch = epoch; job->from = int(decision.from_token); job->to = int(decision.to_token);
      job->layer_count=layers; job->deltas.resize(layers);
      std::fprintf(stderr,"COLD_KV trigger epoch=%llu length=%d from=%d to=%d bytes=%zu threshold=%llu\n",
          (unsigned long long)epoch,cache.length(),job->from,job->to,cache.allocated_payload_bytes(),
          (unsigned long long)policy->effectiveHighBytes());
    }
    if (cache.cold_prefix_length() != job->from || cache.length() < job->to || job->epoch != epoch)
      throw std::runtime_error("cold KV owner changed without lifecycle reset");
    if (job->inflight) {
      const auto result = encoder->poll();
      if (result.state == ColdKVEncoder::State::Failed) { job->inflight=false; std::rethrow_exception(result.error); }
      if (result.state == ColdKVEncoder::State::Ready) {
        job->inflight=false; // TAKE completed even if validation/allocation below throws.
        const auto begin = Clock::now(); const auto& r = *result.result;
        const int expected = job->completed;
        if (r.identity.request_epoch != epoch || r.identity.block_id != uint64_t(expected) ||
            r.identity.layer != expected/job->per_layer() ||
            r.identity.start_position != job->from+(expected%job->per_layer())*256)
          throw std::runtime_error("cold KV stale/out-of-order codec result");
        auto& d = job->deltas[size_t(r.identity.layer)]; const int count = job->to-job->from;
        if (d.k.empty()) { d.k.resize(size_t(16)*count*32); d.v.resize(d.k.size());
          d.ks.resize(size_t(16)*count*2); d.kb.resize(d.ks.size());
          d.vs.resize(d.ks.size()); d.vb.resize(d.ks.size()); }
        max_delta_bytes=std::max(max_delta_bytes,uint64_t(delta_bytes()));
        const int offset = r.identity.start_position-job->from;
        for (int head=0;head<16;++head) {
          const size_t dst=size_t(head)*count+offset, src=size_t(head)*256;
          std::memcpy(d.k.data()+dst*32,r.keys.data.data()+src*32,256*32*sizeof(uint32_t));
          std::memcpy(d.v.data()+dst*32,r.values.data.data()+src*32,256*32*sizeof(uint32_t));
          std::memcpy(d.ks.data()+dst*2,r.keys.scales.data()+src*2,256*2*sizeof(uint16_t));
          std::memcpy(d.kb.data()+dst*2,r.keys.biases.data()+src*2,256*2*sizeof(uint16_t));
          std::memcpy(d.vs.data()+dst*2,r.values.scales.data()+src*2,256*2*sizeof(uint16_t));
          std::memcpy(d.vb.data()+dst*2,r.values.biases.data()+src*2,256*2*sizeof(uint16_t));
        }
        encode_ms += r.encode_ms;
        predict_ms += r.key_timing.predict_ms+r.value_timing.predict_ms;
        pack_ms += r.key_timing.input_copy_ms+r.value_timing.input_copy_ms+
            r.key_timing.output_read_pack_ms+r.value_timing.output_read_pack_ms;
        ++blocks; ++job->completed; job->inflight=false;
        const auto ms=elapsed(begin); collect_ms+=ms; job->owner_ms+=ms;
      } else { ++ticks_while_encoding; return; }
    }
    if (job->completed == job->total()) {
      const auto begin=Clock::now(); drain(); cache.synchronize_storage();
      const auto stream=mx::default_stream(mx::Device::gpu);
      before_bytes=cache.allocated_payload_bytes(); active_before=mx::get_active_memory();
      std::vector<ColdLayer> candidate;
      const int n=job->to-job->from;
      for (int l=0;l<layers;++l) {
        auto& d=job->deltas[size_t(l)];
        auto k=mx::array(d.k.data(),{1,16,n,32},mx::uint32);
        auto v=mx::array(d.v.data(),{1,16,n,32},mx::uint32);
        // uint16 vectors hold IEEE FP16 bits, not uint16 numeric values.
        auto half_array=[&](const auto& data) {
          std::vector<mx::float16_t> typed(data.size());
          static_assert(sizeof(mx::float16_t)==sizeof(uint16_t));
          std::memcpy(typed.data(),data.data(),data.size()*sizeof(uint16_t));
          return mx::array(typed.data(),{1,16,n,2},mx::float16);
        };
        auto ks=half_array(d.ks), kb=half_array(d.kb), vs=half_array(d.vs), vb=half_array(d.vb);
        if (job->from) { const auto& old=cache.cold_layer(l);
          k=mx::concatenate({old.key_codes,k},2,stream); v=mx::concatenate({old.value_codes,v},2,stream);
          ks=mx::concatenate({old.key_scales,ks},2,stream); vs=mx::concatenate({old.value_scales,vs},2,stream);
          kb=mx::concatenate({old.key_biases,kb},2,stream); vb=mx::concatenate({old.value_biases,vb},2,stream); }
        candidate.push_back({std::move(k),std::move(ks),std::move(kb),std::move(v),std::move(vs),std::move(vb),job->to});
        // Pointer constructors copy; release host layer payload as it is staged.
        d=Delta{};
      }
      cache.install_cold_history(std::move(candidate),job->to,stream);
      mx::clear_cache(); cold_tokens=job->to;
      after_bytes=cache.allocated_payload_bytes(); active_after=mx::get_active_memory();
      const auto ms=elapsed(begin); publish_ms+=ms; job->owner_ms+=ms;
      transaction_ms+=elapsed(job->began); ++publications;
      last_completed_position = cache.length(); ++request_publications;
      // Adapt to measured owner interference, not time hidden behind decode.
      policy->published(after_bytes,cold_tokens,before_bytes,job->owner_ms);
      std::fprintf(stderr,"COLD_KV published tokens=%d length=%d blocks=%d before=%zu after=%zu wall_ms=%.3f owner_ms=%.3f snapshot_wait_ms=%.3f publish_ms=%.3f\n",
          cold_tokens,cache.length(),blocks,before_bytes,after_bytes,elapsed(job->began),job->owner_ms,snapshot_wait_ms,ms);
      job.reset(); return;
    }
    if (!supply) return;
    if (job->snapshots.empty() && job->next < job->total()) {
      const auto begin=Clock::now();
      for (int b=0;b<snapshot_batch && job->next<job->total();++b) {
        auto snap=std::make_shared<ColdKVEncoder::Snapshot>(); const int id=job->next;
        snap->identity={epoch,uint64_t(id),id/job->per_layer(),job->from+(id%job->per_layer())*256};
        bool ready=cache.try_copy_frozen_block(snap->identity.layer,snap->identity.start_position,256,snap->keys,snap->values);
        if (!ready && job->snapshots.empty()) {
          ++snapshot_deferrals;
          if (++job->missed <= defer_ticks) break;
          const auto wait=Clock::now(); drain(); cache.synchronize_storage();
          const double ms=elapsed(wait); snapshot_wait_ms+=ms; ++snapshot_drains;
          ready=cache.try_copy_frozen_block(snap->identity.layer,snap->identity.start_position,256,snap->keys,snap->values);
          if (!ready) throw std::runtime_error("cold KV snapshot parents incomplete after drain");
        }
        if (!ready) break;
        job->missed=0; job->snapshots.push_back(std::move(snap)); ++job->next;
      }
      const double total=elapsed(begin);
      // snapshotMS includes memcpy/allocation and any explicit waiting; the
      // separate wait counter is a subset, not an additional request cost.
      snapshot_ms+=total; job->owner_ms+=total;
      max_snapshot_bytes=std::max(max_snapshot_bytes,uint64_t(job->snapshots.size())*2*ColdKVEncoder::kElements*sizeof(uint16_t));
    }
    if (!job->snapshots.empty()) {
      if (!encoder->trySubmit(job->snapshots.front())) return; // try_lock contention: retain and retry.
      job->snapshots.pop_front(); job->inflight=true;
    }
  }
  ~ColdKVInstaller() {
    // Worker target pointers borrow candidate buffers. Join before members are
    // destroyed; their default reverse order would otherwise release Job first.
    if (encoder) encoder->finish();
    job.reset();
  }
  size_t candidate_bytes() const {
    if (!job || !job->candidate) return 0;
    const auto& c=*job->candidate;
    return c.key_codes.nbytes()+c.key_scales.nbytes()+c.key_biases.nbytes()+
        c.value_codes.nbytes()+c.value_scales.nbytes()+c.value_biases.nbytes();
  }
  void prepare_candidate(Cache& cache, int layer) {
    namespace mx=mlx::core;
    if (job->candidate) {
      if (job->candidate_layer!=layer) throw std::runtime_error("cold KV candidate layer differs");
      return;
    }
    const auto begin=Clock::now();
    const int to=job->to;
    auto allocate=[&](int width,mx::Dtype dtype) {
      mx::Shape shape{1,16,to,width};
      auto buffer=mx::allocator::malloc(size_t(16)*to*width*mx::size_of(dtype));
      try {
        return mx::array(buffer,std::move(shape),dtype);
      } catch (...) {
        // The pinned MLX constructor can throw before adopting this non-owning
        // Buffer wrapper; after adoption its remaining initialization is noexcept.
        mx::allocator::free(buffer);
        throw;
      }
    };
    // Only the current layer's final buffers exist. They report available to
    // MLX immediately but must not be submitted to GPU until all blocks finish.
    job->candidate=std::make_unique<ColdLayer>(ColdLayer{
        allocate(32,mx::uint32),allocate(2,mx::float16),allocate(2,mx::float16),
        allocate(32,mx::uint32),allocate(2,mx::float16),allocate(2,mx::float16),to});
    job->candidate_layer=layer;
    max_candidate_bytes=std::max(max_candidate_bytes,uint64_t(candidate_bytes()));
    if (job->from) {
      const auto& old=cache.cold_layer(layer); auto& dest=*job->candidate;
      if (old.tokens!=job->from) throw std::runtime_error("cold KV old prefix changed");
      auto copy=[&](const auto& src,auto& dst,int width) {
        if (!src.is_available()) throw std::runtime_error("cold KV old immutable prefix not ready");
        for(int h=0;h<16;++h) {
          const size_t row=size_t(job->from)*width*src.itemsize();
          std::memcpy(dst.template data<char>()+size_t(h)*to*width*src.itemsize(),
              src.template data<char>()+size_t(h)*row,row);
          old_prefix_copy_bytes+=row;
        }
      };
      copy(old.key_codes,dest.key_codes,32); copy(old.value_codes,dest.value_codes,32);
      copy(old.key_scales,dest.key_scales,2); copy(old.key_biases,dest.key_biases,2);
      copy(old.value_scales,dest.value_scales,2); copy(old.value_biases,dest.value_biases,2);
    }
    const auto ms=elapsed(begin); collect_ms+=ms; job->owner_ms+=ms;
  }
  std::shared_ptr<const ColdKVEncoder::OutputTarget> output_target(const ColdKVEncoder::Identity& id) {
    if (!job->candidate || id.layer!=job->candidate_layer || id.start_position<job->from ||
        id.start_position+256>job->to) throw std::runtime_error("cold KV direct output geometry differs");
    auto output=std::make_shared<ColdKVEncoder::OutputTarget>();output->identity=id;
    const size_t to=size_t(job->to),offset=size_t(id.start_position);
    auto target=[&](auto& codes,auto& scales,auto& biases) {
      ColdKVEncoder::QuantizedTarget t;
      t.data=codes.template data<uint32_t>()+offset*32;
      t.scales=scales.template data<char>()+offset*2*sizeof(uint16_t);
      t.biases=biases.template data<char>()+offset*2*sizeof(uint16_t);
      t.data_capacity_words=size_t(16)*to*32-offset*32;
      t.scales_capacity_groups=t.biases_capacity_groups=size_t(16)*to*2-offset*2;
      t.vector_stride_words=32;t.vector_stride_groups=2;
      t.head_stride_words=to*32;t.head_stride_groups=to*2;
      return t;
    };
    auto& c=*job->candidate;
    output->keys=target(c.key_codes,c.key_scales,c.key_biases);
    output->values=target(c.value_codes,c.value_scales,c.value_biases);
    // No MLX lease is sent to the worker; Job lifetime is owner-confined.
    return output;
  }
  void tick(Cache& cache,const std::function<void()>& drain,bool supply=true) {
    if (!encoder) return;
    if (!layered) {tick_legacy(cache,drain,supply);return;}
    namespace mx=mlx::core;
    if (!job) {
      if (!supply) return;
      const auto decision=policy->evaluate(cache.allocated_payload_bytes(),cache.length(),cache.cold_prefix_length());
      if (!decision.trigger) return;
      last_trigger_position = cache.length();
      job=std::make_unique<Job>();job->epoch=epoch;job->from=int(decision.from_token);
      job->to=int(decision.to_token);job->layer_count=layers;job->original_bytes=cache.allocated_payload_bytes();
      for(int l=0;l<layers;++l) if(cache.layer_cold_prefix_length(l)!=job->from)
        throw std::runtime_error("cold KV new job requires aligned layer prefixes");
      std::fprintf(stderr,"COLD_KV trigger epoch=%llu length=%d from=%d to=%d bytes=%zu threshold=%llu\n",
          (unsigned long long)epoch,cache.length(),job->from,job->to,job->original_bytes,
          (unsigned long long)policy->effectiveHighBytes());
    }
    if (cache.length()<job->to || job->epoch!=epoch)
      throw std::runtime_error("cold KV owner changed without lifecycle reset");
    if (job->inflight) {
      const auto polled=encoder->poll();
      if (polled.state==ColdKVEncoder::State::Failed) {job->inflight=false;std::rethrow_exception(polled.error);}
      if (polled.state!=ColdKVEncoder::State::Ready) {++ticks_while_encoding;return;}
      job->inflight=false;
      const auto begin=Clock::now();const auto& r=*polled.result;const int id=job->completed;
      if (!r.direct_output || !r.keys.data.empty() || !r.values.data.empty() ||
          r.identity.request_epoch!=epoch || r.identity.block_id!=uint64_t(id) ||
          r.identity.layer!=id/job->per_layer() ||
          r.identity.start_position!=job->from+(id%job->per_layer())*256)
        throw std::runtime_error("cold KV direct completion mismatch");
      encode_ms+=r.encode_ms;predict_ms+=r.key_timing.predict_ms+r.value_timing.predict_ms;
      pack_ms+=r.key_timing.input_copy_ms+r.value_timing.input_copy_ms+
          r.key_timing.output_read_pack_ms+r.value_timing.output_read_pack_ms;
      ++blocks;++job->completed;
      const double ms=elapsed(begin);collect_ms+=ms;job->owner_ms+=ms;
      if(job->completed%job->per_layer()==0) {
        if(!job->snapshots.empty() || job->next!=job->completed || !job->candidate)
          throw std::runtime_error("cold KV layer publication has unfinished targets");
        const auto began=Clock::now();drain();cache.synchronize_storage();
        const size_t before=cache.allocated_payload_bytes(); const int layer=job->candidate_layer;
        if(cache.layer_cold_prefix_length(layer)!=job->from)
          throw std::runtime_error("cold KV layer prefix changed before publication");
        active_before=mx::get_active_memory();
        auto completed_candidate=std::move(job->candidate);
        cache.install_cold_layer(layer,std::move(*completed_candidate),mx::default_stream(mx::Device::gpu));
        job->candidate_layer=-1;
        mx::clear_cache();active_after=mx::get_active_memory();
        const double elapsed_ms=elapsed(began);
        publish_ms+=elapsed_ms;job->layer_publish_ms+=elapsed_ms;job->owner_ms+=elapsed_ms;++layer_publications;
        cold_tokens=cache.cold_prefix_length();
        std::fprintf(stderr,"COLD_KV layer_published epoch=%llu layer=%d from=%d tokens=%d length=%d before=%zu after=%zu publish_ms=%.3f\n",
            (unsigned long long)epoch,layer,job->from,job->to,cache.length(),before,cache.allocated_payload_bytes(),elapsed_ms);
      }
    }
    if(job->completed==job->total()) {
      if(job->candidate || job->inflight || !job->snapshots.empty() || cold_tokens!=job->to)
        throw std::runtime_error("cold KV finished job retains candidate or mixed endpoints");
      before_bytes=job->original_bytes;after_bytes=cache.allocated_payload_bytes();
      transaction_ms+=elapsed(job->began);++publications;
      last_completed_position = cache.length(); ++request_publications;
      policy->published(after_bytes,cold_tokens,before_bytes,job->owner_ms);
      std::fprintf(stderr,"COLD_KV published tokens=%d length=%d blocks=%d before=%zu after=%zu wall_ms=%.3f owner_ms=%.3f snapshot_wait_ms=%.3f publish_ms=%.3f\n",
          cold_tokens,cache.length(),blocks,before_bytes,after_bytes,elapsed(job->began),job->owner_ms,snapshot_wait_ms,job->layer_publish_ms);
      job.reset();return;
    }
    if(!supply)return;
    if(job->snapshots.empty() && job->next<job->total()) {
      const int layer=job->next/job->per_layer();
      // Snapshots never cross the current layer: publish/reclaim it before the
      // next layer allocates a candidate or lends any target pointers.
      prepare_candidate(cache,layer);
      const auto began=Clock::now();
      for(int b=0;b<snapshot_batch && job->next<job->total() && job->next/job->per_layer()==layer;++b) {
        auto snap=std::make_shared<ColdKVEncoder::Snapshot>();const int id=job->next;
        snap->identity={epoch,uint64_t(id),layer,job->from+(id%job->per_layer())*256};
        bool ready=cache.try_copy_frozen_block(layer,snap->identity.start_position,256,snap->keys,snap->values);
        if(!ready && job->snapshots.empty()) {
          ++snapshot_deferrals;if(++job->missed<=defer_ticks)break;
          const auto wait=Clock::now();drain();cache.synchronize_storage();
          snapshot_wait_ms+=elapsed(wait);++snapshot_drains;
          ready=cache.try_copy_frozen_block(layer,snap->identity.start_position,256,snap->keys,snap->values);
          if(!ready)throw std::runtime_error("cold KV layer snapshot incomplete after drain");
        }
        if(!ready)break;
        snap->output=output_target(snap->identity);
        job->missed=0;job->snapshots.push_back(std::move(snap));++job->next;
      }
      const double ms=elapsed(began);snapshot_ms+=ms;job->owner_ms+=ms;
      max_snapshot_bytes=std::max(max_snapshot_bytes,uint64_t(job->snapshots.size())*2*ColdKVEncoder::kElements*sizeof(uint16_t));
    }
    if(!job->snapshots.empty()) {
      if(!encoder->trySubmit(job->snapshots.front()))return;
      job->snapshots.pop_front();job->inflight=true;
    }
  }

  // Finish only the block already accepted by the worker. Do not snapshot,
  // encode or initiate any further work. Ready results may publish at the
  // existing owner boundary; an incomplete candidate stays owner-confined.
  // This separates visible-token completion from outstanding background work.
  void drain_accepted(Cache& cache, const std::function<void()>& drain) {
    if (!encoder) return;
    while (job && job->inflight) {
      tick(cache, drain, false);
      if (job && job->inflight) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Only a real cache clear/rewind invalidates a transaction. At most the one
  // accepted block is joined; queued host snapshots are discarded. Never
  // destroy/recreate the model or worker on each request.
  void reset() {
    if (!encoder) return;
    const auto begin=Clock::now();
    if (job) {
      ++discarded_jobs;
      while (job->inflight) {
        const auto r=encoder->poll();
        if (r.state==ColdKVEncoder::State::Failed) { job->inflight=false; std::rethrow_exception(r.error); }
        if (r.state==ColdKVEncoder::State::Ready) job->inflight=false;
        else if (r.state!=ColdKVEncoder::State::Busy) {
          job->inflight=false; throw std::runtime_error("cold KV outstanding result lost during reset");
        } else std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
    job.reset(); ++epoch; cold_tokens=0; policy->reset(); reset_wait_ms+=elapsed(begin);
    last_trigger_position = -1; last_completed_position = -1; request_publications = 0;
  }
};
