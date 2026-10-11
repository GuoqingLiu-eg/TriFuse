#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// One-way, request-scoped cooperative execution permission. Cancellation is
/// nonblocking and never acquires a resident owner or MLX lock. An operation
/// admitted before cancellation may finish; this does not preempt Metal work.
@interface TFETNativeExecutionPermit : NSObject
- (void)cancel NS_SWIFT_NAME(cancel());
@property(nonatomic, readonly, getter=isCancelled) BOOL cancelled;
@end

/// A CPU-only publication endpoint. Background advisers may hold this object
/// without calling the resident owner or waiting for its GPU execution lock.
/// Positions are per KV head, shared across layers in this first adapter.
@interface TFETAttentionSelectionChannel : NSObject
- (instancetype)init NS_UNAVAILABLE;
@property(nonatomic, readonly) uint64_t requestEpoch;
/// NSNumber(false) means stale/invalid proposal; successful publication is not
/// proof of application. All heads require equal, sorted unique prefix counts.
- (nullable NSNumber *)publishPositionsByHead:(NSArray<NSArray<NSNumber *> *> *)positions
                               coveredLength:(NSInteger)coveredLength
                          validUntilPosition:(NSInteger)validUntilPosition
                                requestEpoch:(uint64_t)requestEpoch
                                  proposalID:(uint64_t)proposalID
                                       error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(publish(positionsByHead:coveredLength:validUntilPosition:requestEpoch:proposalID:));
@end

/// Serial B1 floating-point cache, shared by the real ExecuTorch MLX Modules.
/// generation is local to this opaque instance, NOT a shared semantic version.
/// Any dispatched failure invalidates and retains the complete execution
/// context until process exit. There is no retry/reset after unknown completion.
/// Cooperative cancellation is different: known submitted-work drain retires
/// and releases the owner; a failed drain retains it under the same policy.
@interface TFETResidentSession : NSObject
/// Opt-in experimental indexed attention on a fresh full-history GPU owner.
/// Prefill remains dense; T1 requires nativeConfiguration. Epoch changes on
/// reset/rewind/import. Polling never waits for an adviser; no valid proposal
/// means full attention. Backing KV remains complete, but states computed under
/// sparse attention are approximate and are not repaired by later full reads.
- (nullable TFETAttentionSelectionChannel *)enableAttentionSelectionExpectedGeneration:(uint64_t)generation
                                                                               error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(enableAttentionSelection(expectedGeneration:));
/// Experimental hot-view reuse avoids re-gathering a fixed selected prefix on
/// each token; it owns additional KV storage. The original overload uses false.
- (nullable TFETAttentionSelectionChannel *)enableAttentionSelectionReuseHotViews:(BOOL)reuseHotViews
                                                             expectedGeneration:(uint64_t)generation
                                                                          error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(enableAttentionSelection(reuseHotViews:expectedGeneration:));
/// Experimental cross-proposal intersection reuse; requires hot-view reuse.
- (nullable TFETAttentionSelectionChannel *)enableAttentionSelectionReuseHotViews:(BOOL)reuseHotViews
                                                                    deltaReuse:(BOOL)deltaReuse
                                                             expectedGeneration:(uint64_t)generation
                                                                          error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(enableAttentionSelection(reuseHotViews:deltaReuse:expectedGeneration:));
/// Linked runtime capability only; actual operator execution may still fail.
/// No implicit GPU fallback is permitted for an explicit CPU decode request.
+ (BOOL)cpuDecodeAvailable NS_SWIFT_NAME(cpuDecodeAvailable());
/// Standalone CLI only: call before every other MLX operation in that process.
+ (BOOL)configureMetallibPath:(NSString *)path error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(configureMetallibPath(_:));
/// Effective cached math policy and explicit Metal override, not hardware trace.
+ (NSDictionary<NSString *, id> *)runtimeConfiguration NS_SWIFT_NAME(runtimeConfiguration());
/// Coordinator-only restoration of an earlier process-wide idle-cache limit.
/// No model access, GPU wait or cache clear; usable after owner invalidation.
+ (nullable NSDictionary<NSString *, id> *)restoreIdleCacheLimitBytes:(NSNumber *)limit
                                                             error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(restoreIdleCacheLimitBytes(_:));
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:));

/// Optional fixed-Tk verifier shares the same cache as T128 prefill and T1 decode.
/// Pass nil/0 to omit verification; otherwise 0 < verifyTokens <= capacity.
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                verifyPath:(nullable NSString *)verifyPath
                              verifyTokens:(NSInteger)verifyTokens
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:verifyPath:verifyTokens:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:));

/// Optional common external .ptd weights file for every Module. nil preserves
/// inline PTE loading. Each Module owns its loader for the session lifetime;
/// a shared disk file does not establish shared device weight allocations.
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                verifyPath:(nullable NSString *)verifyPath
                              verifyTokens:(NSInteger)verifyTokens
                                  dataPath:(nullable NSString *)dataPath
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:verifyPath:verifyTokens:dataPath:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:));

/// Explicit approximate recent-window policy for a draft. 0 retains full
/// history; a positive window >=128 allocates window+127 physical slots while
/// keeping the original absolute position limit. There are no sink slots.
/// This changes retained draft history, not model weights/RoPE. Ring crops
/// may reject at most127 tokens, whose preceding window is preserved by slack.
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                verifyPath:(nullable NSString *)verifyPath
                              verifyTokens:(NSInteger)verifyTokens
                                  dataPath:(nullable NSString *)dataPath
                               cacheWindow:(NSInteger)cacheWindow
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:verifyPath:verifyTokens:dataPath:cacheWindow:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:));

/// GPU T128 prefill and optional CPU T1 decode bind the same external cache.
/// Device selection is explicit and unavailable CPU execution fails loading;
/// it requires a runtime built with MLX_BUILD_CPU=ON. No application K/V
/// export/import or layout conversion occurs at the phase boundary. Shared
/// storage alone does not prove that kernels avoid internal buffer copies.
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                verifyPath:(nullable NSString *)verifyPath
                              verifyTokens:(NSInteger)verifyTokens
                                  dataPath:(nullable NSString *)dataPath
                               cacheWindow:(NSInteger)cacheWindow
                               decodeOnCPU:(BOOL)decodeOnCPU
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:verifyPath:verifyTokens:dataPath:cacheWindow:decodeOnCPU:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:));

/// Explicit cache storage precision. FP16 programs must expose get_kv_dtype=5
/// in compiled metadata; legacy FP32 programs may omit that method. Output
/// logits remain FP32. Whole-host FP32 snapshots require FP32 storage.
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                verifyPath:(nullable NSString *)verifyPath
                              verifyTokens:(NSInteger)verifyTokens
                                  dataPath:(nullable NSString *)dataPath
                               cacheWindow:(NSInteger)cacheWindow
                               decodeOnCPU:(BOOL)decodeOnCPU
                                   kvDtype:(NSString *)kvDtype
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:verifyPath:verifyTokens:dataPath:cacheWindow:decodeOnCPU:kvDtype:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:));

/// Optional CPU BLAS policy is applied on the actual MLX worker, not the
/// caller thread. nil/default preserves the runtime's existing policy.
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                verifyPath:(nullable NSString *)verifyPath
                              verifyTokens:(NSInteger)verifyTokens
                                  dataPath:(nullable NSString *)dataPath
                               cacheWindow:(NSInteger)cacheWindow
                               decodeOnCPU:(BOOL)decodeOnCPU
                                   kvDtype:(NSString *)kvDtype
                          cpuBLASThreading:(nullable NSString *)cpuBLASThreading
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:verifyPath:verifyTokens:dataPath:cacheWindow:decodeOnCPU:kvDtype:cpuBLASThreading:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:));

/// Fixed prefill shape must match get_prefill_chunk_size and the actual PTE
/// input ABI. Existing initializers retain T128. T512 is full-history, GPU,
/// no-verifier only, with the original8192 bound; cache max_write covers T512.
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                verifyPath:(nullable NSString *)verifyPath
                              verifyTokens:(NSInteger)verifyTokens
                                  dataPath:(nullable NSString *)dataPath
                               cacheWindow:(NSInteger)cacheWindow
                               decodeOnCPU:(BOOL)decodeOnCPU
                                   kvDtype:(NSString *)kvDtype
                          cpuBLASThreading:(nullable NSString *)cpuBLASThreading
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                             prefillTokens:(NSInteger)prefillTokens
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:verifyPath:verifyTokens:dataPath:cacheWindow:decodeOnCPU:kvDtype:cpuBLASThreading:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:prefillTokens:));

/// Explicit request-bound full-history allocation. Zero allocationLimit keeps
/// the legacy compiled-capacity policy. Otherwise 0 < allocationLimit <=
/// capacity caps physical growth and appends, without changing logical model
/// capacity/RoPE or dropping tokens. Optional kvPayloadBudgetBytes must cover
/// paired KV at that ceiling; this is not a process peak-memory limit.
/// Bounded owners require GPU/no verifier and do not support snapshots,
/// retention compaction or feature observers. Reset preserves the limit.
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                verifyPath:(nullable NSString *)verifyPath
                              verifyTokens:(NSInteger)verifyTokens
                                  dataPath:(nullable NSString *)dataPath
                               cacheWindow:(NSInteger)cacheWindow
                               decodeOnCPU:(BOOL)decodeOnCPU
                                   kvDtype:(NSString *)kvDtype
                          cpuBLASThreading:(nullable NSString *)cpuBLASThreading
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                             prefillTokens:(NSInteger)prefillTokens
                           allocationLimit:(NSInteger)allocationLimit
                      kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:verifyPath:verifyTokens:dataPath:cacheWindow:decodeOnCPU:kvDtype:cpuBLASThreading:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:prefillTokens:allocationLimit:kvPayloadBudgetBytes:));

/// Explicit prefill submission queue limit, 1...64. This initializer does not
/// read TRIFUSE_PREFILL_MAX_QUEUED_CHUNKS; malformed environment values cannot
/// override or reject an explicit limit. Older overloads retain environment
/// selection with a default of 4. The limit controls submission/drain cadence,
/// not the model's prefill shape, KV capacity or a process-memory ceiling.
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                verifyPath:(nullable NSString *)verifyPath
                              verifyTokens:(NSInteger)verifyTokens
                                  dataPath:(nullable NSString *)dataPath
                               cacheWindow:(NSInteger)cacheWindow
                               decodeOnCPU:(BOOL)decodeOnCPU
                                   kvDtype:(NSString *)kvDtype
                          cpuBLASThreading:(nullable NSString *)cpuBLASThreading
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                             prefillTokens:(NSInteger)prefillTokens
                           allocationLimit:(NSInteger)allocationLimit
                      kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes
                    prefillMaxQueuedChunks:(NSInteger)prefillMaxQueuedChunks
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:verifyPath:verifyTokens:dataPath:cacheWindow:decodeOnCPU:kvDtype:cpuBLASThreading:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:prefillTokens:allocationLimit:kvPayloadBudgetBytes:prefillMaxQueuedChunks:));

/// Optional native permission for pure full-history GPU target execution.
/// nil retains legacy behavior. Cancellation is observed before module loading
/// and at native operation/chunk boundaries. Interrupted batches retire their
/// owner after a drain of submitted work; unknown completion is quarantined.
- (nullable instancetype)initWithPrefillPath:(NSString *)prefillPath
                                decodePath:(NSString *)decodePath
                                verifyPath:(nullable NSString *)verifyPath
                              verifyTokens:(NSInteger)verifyTokens
                                  dataPath:(nullable NSString *)dataPath
                               cacheWindow:(NSInteger)cacheWindow
                               decodeOnCPU:(BOOL)decodeOnCPU
                                   kvDtype:(NSString *)kvDtype
                          cpuBLASThreading:(nullable NSString *)cpuBLASThreading
                                    layers:(NSInteger)layers
                                   kvHeads:(NSInteger)kvHeads
                                   headDim:(NSInteger)headDim
                                  capacity:(NSInteger)capacity
                                vocabulary:(NSInteger)vocabulary
                        effectiveWeightsID:(NSString *)effectiveWeightsID
                             prefillTokens:(NSInteger)prefillTokens
                           allocationLimit:(NSInteger)allocationLimit
                      kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes
                    prefillMaxQueuedChunks:(NSInteger)prefillMaxQueuedChunks
                     nativeExecutionPermit:(nullable TFETNativeExecutionPermit *)nativeExecutionPermit
                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(init(prefillPath:decodePath:verifyPath:verifyTokens:dataPath:cacheWindow:decodeOnCPU:kvDtype:cpuBLASThreading:layers:kvHeads:headDim:capacity:vocabulary:effectiveWeightsID:prefillTokens:allocationLimit:kvPayloadBudgetBytes:prefillMaxQueuedChunks:nativeExecutionPermit:));

/// Bind the next request's uncancelled permission to an idle compatible owner.
/// An old cancelled permission may be replaced only while the owner is valid.
/// This performs no GPU work and preserves KV, position and generation. A
/// permission cannot be removed through this setter; stale generations reject.
/// A cancelled replacement reports cancelled/not_dispatched without changing
/// the existing owner, its permission, or its cancellation metadata.
- (BOOL)configureNativeExecutionPermit:(TFETNativeExecutionPermit *)permit
                   expectedGeneration:(uint64_t)generation
                                error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(configureNativeExecutionPermit(_:expectedGeneration:));

/// Change only the prefill submission queue limit on a ready owner. Serialized
/// with this owner's other operations; requires matching generation and no
/// pending invocation, prefill batch or import. No GPU operation is performed.
/// Admission errors preserve the owner. Cache, position and generation do not
/// change; inspectMetadata reports the effective prefillMaxQueuedChunks.
- (BOOL)configurePrefillMaxQueuedChunks:(NSInteger)prefillMaxQueuedChunks
                   expectedGeneration:(uint64_t)generation
                                error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(configurePrefill(maxQueuedChunks:expectedGeneration:));

/// Start another independent request on the same loaded GPU modules. This
/// synchronizes and clears old KV, binds a new allocation ceiling/budget, and
/// increments generation without changing sessionID or logical capacity.
/// Zero limit selects compiled capacity. Requires full-history/no verifier/
/// no feature observer. Admission errors preserve the previous request state;
/// unknown completion invalidates the owner. This is not prefix-cache reuse.
- (BOOL)beginRequestWithAllocationLimit:(NSInteger)allocationLimit
                 kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes
                    expectedGeneration:(uint64_t)generation
                                 error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(beginRequest(allocationLimit:kvPayloadBudgetBytes:expectedGeneration:));

/// Retain an exact contiguous prefix while rebinding the next request's KV
/// allocation ceiling. Oversized pools are physically copied and synchronized
/// before publication; pools fitting the ceiling retain their allocation.
/// The budget bounds final owned KV payload, not transient copies or process
/// memory. Admission failures preserve the old owner; unknown completion retires it.
- (BOOL)retainPrefixLength:(NSInteger)length
          allocationLimit:(NSInteger)allocationLimit
     kvPayloadBudgetBytes:(nullable NSNumber *)kvPayloadBudgetBytes
       expectedGeneration:(uint64_t)generation
                    error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(retainPrefix(length:allocationLimit:kvPayloadBudgetBytes:expectedGeneration:));

/// Returns owned FP32 last-token logits. Admission errors do not dispatch.
- (nullable NSData *)forwardTokenIDs:(NSArray<NSNumber *> *)tokens
                          positions:(NSArray<NSNumber *> *)positions
                 expectedGeneration:(uint64_t)generation
                              error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(forward(tokenIDs:positions:expectedGeneration:));

/// Explicit optional native MLX adapter for one synchronous T1 GPU append.
/// Requires flat storage and a configuration matching the actual loaded
/// model/cache owner. Retained diagnostic history permits only this native
/// T1 path. nil retains exported execution for ordinary uncompacted history.
- (nullable NSData *)forwardTokenIDs:(NSArray<NSNumber *> *)tokens
                          positions:(NSArray<NSNumber *> *)positions
                nativeConfiguration:(nullable NSDictionary<NSString *, id> *)configuration
                 expectedGeneration:(uint64_t)generation
                              error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(forward(tokenIDs:positions:nativeConfiguration:expectedGeneration:));

/// Native GPU T1 with device-side greedy selection and finite-logit validation.
/// Consumes tokenID completely, commits one position/generation, then returns
/// the next token as an owned scalar. No lookahead overlap or host logits copy.
/// Full-history owners only; ordinary forward remains the logits/quality API.
- (nullable NSNumber *)forwardGreedyTokenID:(NSNumber *)tokenID
                                 position:(NSNumber *)position
                      nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
                       expectedGeneration:(uint64_t)generation
                                    error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(forwardGreedy(tokenID:position:nativeConfiguration:expectedGeneration:));

/// Internal architecture-aware Granite pipeline. The first token comes from
/// the already completed prefill; subsequent checked predictions stay on GPU.
/// While active, ordinary mutation APIs reject. Finish consumes no extra token,
/// drains every returned token's T1 and commits the completed endpoint once.
- (BOOL)beginGreedyPipelineWithFirstToken:(NSNumber *)firstToken
                    nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
                     expectedGeneration:(uint64_t)generation
                                  error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(beginGreedyPipeline(firstToken:nativeConfiguration:expectedGeneration:));
/// C++ port of Python generate_tokens: fresh prompt[:-1] cache-only T512,
/// queue1/drain/clear1, then prompt-last T1 creates a GPU checked prediction.
/// Returns host admission counters only; no CPU argmax or vocabulary copy.
- (nullable NSDictionary<NSString *, id> *)beginPythonPolicyTokenIDs:(NSArray<NSNumber *> *)tokens
                      nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
                       expectedGeneration:(uint64_t)generation
                                    error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(beginPythonPolicy(tokenIDs:nativeConfiguration:expectedGeneration:));
/// Same C++ prompt traversal for teacher-forced quality measurement.
- (nullable NSData *)prefillPythonPolicyTokenIDs:(NSArray<NSNumber *> *)tokens
                      nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
                       expectedGeneration:(uint64_t)generation
                                    error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(prefillPythonPolicy(tokenIDs:nativeConfiguration:expectedGeneration:));
- (nullable NSNumber *)nextGreedyPipelineExpectedGeneration:(uint64_t)generation
                                                     error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(nextGreedyPipeline(expectedGeneration:));
- (nullable NSDictionary<NSString *, id> *)finishGreedyPipelineExpectedGeneration:(uint64_t)generation
                                                                          error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(finishGreedyPipeline(expectedGeneration:));
- (nullable NSDictionary<NSString *, id> *)finishGreedyPipelineExpectedGeneration:(uint64_t)generation
                                        resumeLastYield:(BOOL)resumeLastYield
                                                 error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(finishGreedyPipeline(expectedGeneration:resumeLastYield:));
/// Host counters only, also available after cancellation/invalid completion.
- (NSDictionary<NSString *, id> *)greedyPipelineReceipt;

/// Wait only for already accepted background maintenance. Never starts another
/// compression block; does not clear KV or change token position/generation.
- (BOOL)drainAcceptedMaintenanceExpectedGeneration:(uint64_t)generation
    error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(drainAcceptedMaintenance(expectedGeneration:));

/// Offline dense T1 label collection. Returns owned logits and small FP32
/// block scores recomputed from native Q/K, not fused-kernel probabilities.
/// No adviser/feature observer/compacted history may be attached.
- (nullable NSDictionary<NSString *, id> *)forwardWithAttentionLabelTokenIDs:(NSArray<NSNumber *> *)tokens
                                                                positions:(NSArray<NSNumber *> *)positions
                                                      nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
                                                                   layers:(NSArray<NSNumber *> *)layers
                                                       expectedGeneration:(uint64_t)generation
                                                                    error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(forwardWithAttentionLabels(tokenIDs:positions:nativeConfiguration:layers:expectedGeneration:));

/// Bounded asynchronous fixed-shape prefill (T128 or admitted T512).
/// At most four chunks are queued before a drain. Intermediate chunks submit only KV
/// roots; output bytes are read only from the final synchronous chunk. All
/// inputs remain owned until completion. One committed generation is published
/// after the entire batch drains; a dispatched failure invalidates the owner.
- (nullable NSData *)prefillTokenIDs:(NSArray<NSNumber *> *)tokens
                    expectedGeneration:(uint64_t)generation
                                 error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(prefill(tokenIDs:expectedGeneration:));

/// Explicit standard-Llama adapter using native MLX graph construction with
/// the same loaded constants and cache owner. nil keeps the exported graph.
/// A nonnil configuration accepts any positive token count within capacity.
/// The fixed ABI's final short block is zero-padded only for transport; native
/// computation and KV include the real prefix only. Returns the last real
/// token's logits and publishes the actual length once the entire batch drains.
/// Feature observation still requires whole blocks and fails before dispatch.
- (nullable NSData *)prefillTokenIDs:(NSArray<NSNumber *> *)tokens
                nativeConfiguration:(nullable NSDictionary<NSString *, id> *)configuration
                 expectedGeneration:(uint64_t)generation
                              error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(prefill(tokenIDs:nativeConfiguration:expectedGeneration:));

/// Optional, advisory-only prefill observation. Configuration contains sorted
/// layer indices, block_tokens, and max_pending (1..8). Install once on a fresh
/// owner. Consumer runs on a separate serial queue with owned small FP32 data;
/// it must not call this owner. No KV action or historical KV read is performed.
/// Queue pressure skips collection before constructing the summary graph.
- (BOOL)configurePrefillFeatures:(NSDictionary<NSString *, id> *)configuration
                       consumer:(void (^)(NSDictionary<NSString *, id> *batch))consumer
             expectedGeneration:(uint64_t)generation
                          error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(configurePrefillFeatures(_:consumer:expectedGeneration:));

/// Join the advisory queue after model execution, outside the decode loop.
/// Its wait cost belongs in the complete request. Includes observation errors;
/// the caller must fail the observation if failed_batches is nonzero. It does
/// not imply that a policy was trained, ANE ran, or KV memory was reduced.
- (NSDictionary<NSString *, id> *)drainPrefillFeatures;

/// Explicit process-wide idle-buffer maintenance at a quiescent CPU scheduling
/// boundary. Caller must stop every model/observer and call from a thread that
/// can access all registered MLX streams. Foreign private streams fail closed.
/// This owner must have no observer or outstanding invocation. Live KV and its
/// position/generation are unchanged. nil preserves the idle-cache limit;
/// a nonnegative byte limit persists process-wide until explicitly changed.
/// The returned previous limit permits restoration by the coordinator. This is
/// not a cap on active/process memory, and no OS reclamation is promised.
- (nullable NSDictionary<NSString *, id> *)reclaimIdleMemoryWithCacheLimitBytes:(nullable NSNumber *)limit
                                                         expectedGeneration:(uint64_t)generation
                                                                      error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(reclaimIdleMemory(cacheLimitBytes:expectedGeneration:));

/// Diagnostic-only physical retention after a completed native prefill and
/// before any T1/crop/import/reset. Requires GPU, no verifier, flat storage and
/// no feature observer. Positions are sorted unique original absolute slots.
/// Configuration must match the adapter that actually completed prefill.
/// Success keeps the absolute next position and advances generation. Receipt
/// bytes describe owned current KV payload/backing buffers, not process RSS or
/// allocator memory returned to the system. Subsequent appends require native
/// T1; known admission failures do not dispatch, later failures invalidate.
- (nullable NSDictionary<NSString *, id> *)compactRetainingPositions:(NSArray<NSNumber *> *)positions
                                               nativeConfiguration:(NSDictionary<NSString *, id> *)configuration
                                                expectedGeneration:(uint64_t)generation
                                                             error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(compactRetainingPositions(_:nativeConfiguration:expectedGeneration:));

/// Exactly Tk contiguous new positions; returns owned row-major [1,Tk,V]
/// FP32 logits. Every input token is committed before return, including draft
/// tokens later rejected by the caller. No implicit sampling or acceptance.
- (nullable NSData *)verifyTokenIDs:(NSArray<NSNumber *> *)tokens
                         positions:(NSArray<NSNumber *> *)positions
                expectedGeneration:(uint64_t)generation
                             error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(verify(tokenIDs:positions:expectedGeneration:));

/// Crop to an absolute committed length, synchronously, without copying K/V.
/// Publication changes logical length only: no GPU dispatch or physical cleanup.
/// Rejected slots remain allocated but are outside readable history. A ring
/// requires <=127 rejected tokens; its slack preserves the preceding window.
/// Advances generation even for an unchanged
/// length. Completion errors invalidate the session; they must not be retried.
- (BOOL)acceptPrefixLength:(NSInteger)length
       expectedGeneration:(uint64_t)generation
                    error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(acceptPrefix(length:expectedGeneration:));

/// Metadata plus process-wide MLX allocator counters. Call only when all model
/// execution/allocation is quiescent; the allocator counters are not atomic.
/// Also available for invalid sessions.
- (nullable NSDictionary<NSString *, id> *)inspect:(NSError * _Nullable * _Nullable)error;

/// Owner metadata only, protected by this session's mutex. Does not read global
/// allocator counters or dispatch/synchronize GPU work. Use for asynchronous
/// logical crop receipts while an independent draft model executes.
- (nullable NSDictionary<NSString *, id> *)inspectMetadata:(NSError * _Nullable * _Nullable)error;

/// Whole K/V host snapshot with shape, validLength, local generation,
/// effectiveWeightsID and layout=flat_bhcd_fp32. Future bytes are preserved.
- (nullable NSDictionary<NSString *, id> *)exportStateExpectedGeneration:(uint64_t)generation
    error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(exportState(expectedGeneration:));

/// Validate/copy both arrays before dispatch, then publish the pair atomically.
/// Input must contain keys/values NSData, shape, validLength, effectiveWeightsID,
/// and layout. Native generation is advanced independently of source versions.
- (BOOL)importState:(NSDictionary<NSString *, id> *)state
    expectedGeneration:(uint64_t)generation error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(importState(_:expectedGeneration:));
- (BOOL)resetExpectedGeneration:(uint64_t)generation
    error:(NSError * _Nullable * _Nullable)error
    NS_SWIFT_NAME(reset(expectedGeneration:));

/// If Swift postconditions fail, discard this actual cache as well as metadata.
- (void)invalidateWithReason:(NSString *)reason NS_SWIFT_NAME(invalidate(reason:));
@end
NS_ASSUME_NONNULL_END
