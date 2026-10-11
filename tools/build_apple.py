#!/usr/bin/env python3
"""Build the Apple resident bridge and its pinned ExecuTorch/MLX dependencies.

--plan and --dry-run only inspect paths/tools and print a JSON command plan;
missing dependencies are reported without making those modes fail. Real builds
require bootstrapped source, Xcode with Metal tools, CMake/Ninja, and the Python
code-generation dependencies. No model, signing, installation or device access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Relative paths are the outputs of the existing single-config CMake targets.
LIBRARIES = {
    'executorch': 'libexecutorch.a',
    'executorch_core': 'libexecutorch_core.a',
    'extension_module': 'extension/module/libextension_module.a',
    'extension_tensor': 'extension/tensor/libextension_tensor.a',
    'extension_data_loader': 'extension/data_loader/libextension_data_loader.a',
    'extension_flat_tensor': 'extension/flat_tensor/libextension_flat_tensor.a',
    'extension_named_data_map': 'extension/named_data_map/libextension_named_data_map.a',
    'portable_ops_lib': 'kernels/portable/libportable_ops_lib.a',
    'portable_kernels': 'kernels/portable/libportable_kernels.a',
    'kernels_util_all_deps': 'kernels/portable/cpu/util/libkernels_util_all_deps.a',
    'extension_llm_cache': 'extension/llm/cache/libextension_llm_cache.a',
    'mlxdelegate': 'backends/mlx/libmlxdelegate.a',
}
FORCE_LOAD = {'executorch', 'portable_ops_lib', 'mlxdelegate'}
SOURCE_REQUIREMENTS = (
    'CMakeLists.txt', 'tools/cmake/Utils.cmake',
    'third-party/ios-cmake/ios.toolchain.cmake',
    'third-party/flatbuffers/CMakeLists.txt', 'third-party/flatcc/CMakeLists.txt',
    'third-party/gflags/CMakeLists.txt', 'third-party/json/CMakeLists.txt',
    'third-party/pocketfft/pocketfft_hdronly.h',
    'backends/xnnpack/third-party/FXdiv/CMakeLists.txt',
    'backends/mlx/CMakeLists.txt', 'backends/mlx/patches/apply.sh',
    'backends/mlx/serialization/schema.fbs',
    'backends/mlx/serialization/generate.py',
    'backends/mlx/serialization/MLXLoader.h.tmpl',
    'backends/mlx/serialization/MLXLoader.cpp.tmpl',
    'backends/mlx/third-party/mlx/CMakeLists.txt',
    'extension/llm/cache/CMakeLists.txt',
    'extension/llm/tokenizers/CMakeLists.txt',
    'extension/llm/tokenizers/third-party/abseil-cpp/CMakeLists.txt',
    'extension/llm/tokenizers/third-party/re2/CMakeLists.txt',
    'extension/llm/tokenizers/third-party/sentencepiece/CMakeLists.txt',
    'extension/llm/tokenizers/third-party/json/single_include/nlohmann/json.hpp',
    'extension/llm/tokenizers/third-party/llama.cpp-unicode/include/unicode.h',
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_plan(args):
    source = ROOT / 'trifuse/executorch'
    build = args.build_dir.resolve() if args.build_dir else ROOT / 'build' / args.platform
    cmake_build, objects, output = build / 'executorch', build / 'objects', build / 'package'
    sdk = 'iphoneos' if args.platform == 'ios' else 'macosx'
    cmake_platform = 'OS64' if args.platform == 'ios' else 'MAC_ARM64'
    target = 'arm64-apple-' + ('ios' if args.platform == 'ios' else 'macos') + args.deployment_target
    # Keep a venv interpreter's symlink path: resolving it would select the base
    # interpreter and lose the venv's site-packages and build-tool executables.
    python = os.path.abspath(os.path.expanduser(args.python)) if '/' in args.python else (shutil.which(args.python) or args.python)
    sources = [ROOT / 'src/apple/TFETResidentSession.mm']
    if args.cold_kv:
        sources.append(ROOT / 'src/apple/ColdKVEncoder.mm')
    required = [source / path for path in SOURCE_REQUIREMENTS] + sources + [
        ROOT / 'src/apple/TFETSharedSegmentDataLoader.h',
        ROOT / 'include/trifuse/apple/TFETResidentSession.h',
        ROOT / 'include/trifuse/kv/ColdKVAttention.h',
    ]
    if args.cold_kv:
        required += [ROOT / 'include/trifuse/kv' / name for name in
                     ('ColdKVInstaller.h', 'ColdKVEncoder.h', 'ThresholdPolicy.h')]
    errors = ['missing source dependency: ' + str(path) for path in required if not path.is_file()]
    tool_path = str(Path(python).parent) + os.pathsep + os.environ.get('PATH', '')
    tools = {name: shutil.which(name, path=tool_path) for name in ('cmake', 'ninja', 'xcrun')}
    errors += ['missing executable: ' + name for name, path in tools.items() if path is None]
    if not Path(python).is_file():
        errors.append('Python interpreter not found: ' + python)
    if platform.system() != 'Darwin' or platform.machine() not in ('arm64', 'aarch64'):
        errors.append('Real builds require an Apple Silicon macOS host')
    flags = {
        'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_CXX_STANDARD': '20',
        'CMAKE_TOOLCHAIN_FILE': str(source / 'third-party/ios-cmake/ios.toolchain.cmake'),
        'PLATFORM': cmake_platform, 'DEPLOYMENT_TARGET': args.deployment_target,
        'CMAKE_OSX_DEPLOYMENT_TARGET': args.deployment_target,
        'CMAKE_CXX_FLAGS': '-I' + shlex.quote(str(ROOT / 'include')),
        'PYTHON_EXECUTABLE': python, 'Python_EXECUTABLE': python, 'Python3_EXECUTABLE': python,
        'BUILD_SHARED_LIBS': 'OFF', 'BUILD_TESTING': 'OFF',
        'EXECUTORCH_BUILD_SHARED': 'OFF', 'EXECUTORCH_BUILD_TESTS': 'OFF',
        'EXECUTORCH_BUILD_PYBIND': 'OFF', 'EXECUTORCH_BUILD_EXECUTOR_RUNNER': 'OFF',
        'EXECUTORCH_BUILD_KERNELS_CUSTOM_AOT': 'OFF', 'EXECUTORCH_BUILD_PORTABLE_OPS': 'ON',
        'EXECUTORCH_BUILD_EXTENSION_MODULE': 'ON', 'EXECUTORCH_BUILD_EXTENSION_TENSOR': 'ON',
        'EXECUTORCH_BUILD_EXTENSION_DATA_LOADER': 'ON', 'EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR': 'ON',
        'EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP': 'ON', 'EXECUTORCH_BUILD_EXTENSION_LLM': 'ON',
        'EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER': 'OFF',
        'EXECUTORCH_BUILD_MLX': 'ON', 'EXECUTORCH_MLX_ENABLE_CPU': 'OFF',
        'EXECUTORCH_BUILD_COREML': 'OFF', 'EXECUTORCH_BUILD_XNNPACK': 'OFF',
        'EXECUTORCH_BUILD_CPUINFO': 'OFF', 'EXECUTORCH_BUILD_PTHREADPOOL': 'OFF',
    }
    if args.platform == 'macos':
        # ios-cmake otherwise marks auxiliary command-line targets as bundles,
        # which breaks SentencePiece's ordinary executable install rules.
        flags['CMAKE_MACOSX_BUNDLE'] = 'OFF'
    configure = ['cmake', '-S', str(source), '-B', str(cmake_build), '-G', 'Ninja']
    configure += ['-D' + key + '=' + value for key, value in flags.items()]
    steps = [
        {'kind': 'python_codegen_check', 'argv': [python, '-c',
          "import importlib.util; missing=[x for x in ('torchgen','yaml','flatbuffers') if importlib.util.find_spec(x) is None]; assert not missing, 'Missing codegen dependencies: '+', '.join(missing)"]},
        {'kind': 'metal_toolchain_check', 'argv': ['xcrun', '--sdk', sdk, 'metal', '--version']},
        {'kind': 'sdk_query', 'argv': ['xcrun', '--sdk', sdk, '--show-sdk-path'], 'sets': '@SDKROOT@'},
        {'kind': 'configure', 'argv': configure},
        {'kind': 'build_dependencies', 'argv': ['cmake', '--build', str(cmake_build), '--target',
            *LIBRARIES, '--parallel', str(args.jobs)]},
    ]
    includes = [ROOT / 'include', ROOT / 'include/trifuse/apple', ROOT / 'include/trifuse/kv',
        ROOT / 'src/apple', ROOT / 'trifuse',
        source / 'runtime/core/portable_type/c10', cmake_build / 'include',
        cmake_build / 'extension/flat_tensor/include', source / 'third-party/flatbuffers/include',
        source / 'backends/mlx/third-party/mlx']
    compiler = ['xcrun', '--sdk', sdk, 'clang++', '-std=c++20', '-O2', '-fobjc-arc', '-fexceptions',
                '-target', target, '-isysroot', '@SDKROOT@', '-DC10_USING_CUSTOM_GENERATED_MACROS']
    if args.cold_kv:
        compiler.append('-DTRIFUSE_ENABLE_COLD_KV')
    for path in includes:
        compiler += ['-I', str(path)]
    object_paths = []
    for path in sources:
        obj = objects / (path.stem + '.o')
        object_paths.append(obj)
        steps.append({'kind': 'compile_bridge', 'argv': compiler + ['-c', str(path), '-o', str(obj)]})
    archive = output / 'lib/libTriFuse.a'
    steps.append({'kind': 'archive_bridge', 'argv': ['xcrun', 'libtool', '-static', '-o', str(archive), *map(str, object_paths)]})
    libraries = {name: str(cmake_build / relative) for name, relative in LIBRARIES.items()}
    libraries['mlx'] = str(cmake_build / 'backends/mlx/mlx/libmlx.a')
    packaged = {name: str(output / 'lib' / Path(path).name) for name, path in libraries.items()}
    framework_names = ['Foundation', 'Metal', 'QuartzCore', 'Accelerate']
    if args.cold_kv:
        framework_names.append('CoreML')
    link_args = [str(archive)]
    for name, path in packaged.items():
        link_args += ['-Xlinker', '-force_load', '-Xlinker', path] if name in FORCE_LOAD else [path]
    for name in framework_names:
        link_args += ['-framework', name]
    link_args.append('-lc++')
    metallib = cmake_build / 'backends/mlx/mlx/mlx/backend/metal/kernels/mlx.metallib'
    bundle = output / 'resources/executorch_backend_mlx_resources.bundle'
    resource_name = 'mlx-' + args.platform + '.metallib'
    steps.append({'kind': 'package', 'libraries': libraries, 'destination': str(output),
                  'metallib': str(metallib), 'resource_bundle': str(bundle), 'resource_name': resource_name})
    steps.append({'kind': 'link_only_check', 'argv': compiler + [str(objects / 'link_check.mm'),
        *link_args, '-o', str(build / 'link-check')], 'executable_is_never_run': True})
    return {'schema': 'trifuse.apple_source_build.v1', 'platform': args.platform,
        'target': target, 'cold_kv_compiled': args.cold_kv, 'build_directory': str(build),
        'package_directory': str(output), 'python': python, 'tool_paths': tools, 'jobs': args.jobs,
        'dependency_errors': errors, 'ready_for_build_preflight': not errors,
        'commands': steps, 'consumer_link_arguments': link_args,
        'public_header': str(output / 'include/trifuse/apple/TFETResidentSession.h'),
        'consumer_include_directory': str(output / 'include'),
        'resources_to_embed': [str(bundle)], 'model_executed': False,
        'notes': [
            'Run tools/bootstrap.py --submodules first; initialize the nested tokenizer dependencies listed in dependency_errors.',
            'CMake builds host flatc/flatcc from source and may fetch MLX source dependencies; no pretrained model is fetched.',
            'Python codegen imports and Metal SDK availability are checked only during a real build, before compilation.',
            'The SDKROOT token in the plan is replaced by the sdk_query result; no shell expansion is used.',
            'Copy the resource bundle into the consuming App; a command-line consumer can call configureMetallibPath before MLX use.',
            'Cold KV needs a separately generated affine_bounds CoreML package at runtime, including in CPU_ONLY mode.',
            'This produces arm64 static libraries and a link check, not a signed App or an XCFramework.',
        ]}


def execute(plan):
    if plan['dependency_errors']:
        raise RuntimeError('\n'.join(plan['dependency_errors']))
    build, output = Path(plan['build_directory']), Path(plan['package_directory'])
    # Reuse CMake/object outputs for incremental builds, but never leave a stale
    # successful package/manifest mixed with a different build configuration.
    incomplete = output / '.trifuse-incomplete'
    if output.exists():
        if not incomplete.is_file() or (output / 'build-manifest.json').exists():
            raise RuntimeError('Package output already exists; choose a fresh --build-dir: ' + str(output))
        # Only discard staging output explicitly marked by our own failed run.
        # Keep dependency/object builds so a retry does not duplicate them.
        shutil.rmtree(output)
    objects = build / 'objects'
    objects.mkdir(parents=True, exist_ok=True)
    (output / 'lib').mkdir(parents=True)
    incomplete.write_text('Incomplete TriFuse build output; safe to replace on retry.\n')
    sdkroot = None
    environment = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=str(plan['jobs']),
        PATH=str(Path(plan['python']).parent) + os.pathsep + os.environ.get('PATH', ''))
    for step in plan['commands']:
        if step['kind'] == 'package':
            for path in step['libraries'].values():
                shutil.copy2(path, output / 'lib' / Path(path).name)
            shutil.copytree(ROOT / 'include/trifuse', output / 'include/trifuse')
            shutil.copy2(ROOT / 'include/module.modulemap', output / 'include/module.modulemap')
            bundle = Path(step['resource_bundle'])
            bundle.mkdir(parents=True)
            shutil.copy2(step['metallib'], bundle / step['resource_name'])
            (bundle / 'Info.plist').write_bytes(plistlib.dumps({
                'CFBundleIdentifier': 'org.pytorch.executorch.backend-mlx-resources',
                'CFBundleName': 'executorch_backend_mlx_resources',
                'CFBundlePackageType': 'BNDL', 'CFBundleVersion': '1'}))
            continue
        if step['kind'] == 'link_only_check':
            (objects / 'link_check.mm').write_text('#import "TFETResidentSession.h"\nint main() { return [TFETResidentSession class] ? 0 : 1; }\n')
        command = [sdkroot if arg == '@SDKROOT@' else arg for arg in step['argv']]
        if step['kind'] == 'sdk_query':
            sdkroot = subprocess.check_output(command, text=True, env=environment).strip()
            if not Path(sdkroot).is_dir():
                raise RuntimeError('xcrun returned an invalid SDK path')
        else:
            print('+ ' + shlex.join(command), flush=True)
            subprocess.run(command, check=True, env=environment)
    plan.update(status='built_and_linked', sdk_path=sdkroot, build_executed=True,
        package_sha256={str(path.relative_to(output)): sha(path) for path in sorted(output.rglob('*'))
                        if path.is_file() and path != incomplete})
    manifest_staging = output / '.build-manifest.json.tmp'
    manifest_staging.write_text(json.dumps(plan, indent=2) + '\n')
    manifest_staging.replace(output / 'build-manifest.json')
    incomplete.unlink()
    print(json.dumps({'status': plan['status'], 'package': str(output), 'model_executed': False}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=('ios', 'macos'), required=True)
    parser.add_argument('--build-dir', type=Path, help='Defaults to build/<platform>; use a fresh package destination')
    parser.add_argument('--python', default=sys.executable, help='Python with torchgen, PyYAML and flatbuffers')
    parser.add_argument('--deployment-target', default='26.2')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--no-cold-kv', dest='cold_kv', action='store_false', default=True)
    parser.add_argument('--plan', '--dry-run', dest='dry_run', action='store_true')
    args = parser.parse_args(argv)
    if args.jobs < 1 or not re.fullmatch(r'\d+\.\d+(?:\.\d+)?', args.deployment_target):
        parser.error('jobs must be positive and deployment-target must be a numeric version')
    plan = make_plan(args)
    if args.dry_run:
        plan.update(status='plan_only', build_executed=False)
        print(json.dumps(plan, indent=2))
        return 0
    try:
        execute(plan)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print('Build failed: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
