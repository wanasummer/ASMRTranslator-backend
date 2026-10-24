#!/usr/bin/env node

/**
 * Build the production Python sidecar with Nuitka.
 *
 * Output:
 *   build_staging_elec/resources/main.exe
 *
 * This is the backend packaging step used by the Tauri installer workflow.
 */

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const OUTPUT_DIR = path.join(ROOT, "build_nuitka");
const RESOURCE_DIR = path.join(ROOT, "build_staging_elec", "resources");
const BUILT_EXE = path.join(OUTPUT_DIR, "main.exe");
const SIDECAR = path.join(RESOURCE_DIR, "main.exe");
const PACKAGE_CONFIG = path.join(ROOT, "nuitka-package.config.yml");
const GGUF = path.join(ROOT, "modules", "vad", "model", "fsmn-vad.gguf");
const VAD_DLL = path.join(ROOT, "modules", "vad", "native", "fsmn_vad.dll");
const UNIDIC_DIR = path.join(
  ROOT, ".venv", "Lib", "site-packages", "unidic_lite", "dicdir"
);
const FAST_BUILD = /^(1|true|yes)$/i.test(process.env.ASMR_FAST_BUILD || "");

function run(command, args, options = {}) {
  console.log(`\n> ${command} ${args.join(" ")}`);
  const executable = process.platform === "win32" && command === "uv" ? "uv.exe" : command;
  const result = spawnSync(executable, args, {
    cwd: ROOT,
    stdio: "inherit",
    ...options,
  });

  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

function removeDirectory(directory) {
  if (!fs.existsSync(directory)) return;
  fs.rmSync(directory, { recursive: true, force: true, maxRetries: 3 });
}

function requireFile(filename, label) {
  if (!fs.existsSync(filename) || !fs.statSync(filename).isFile()) {
    console.error(`${label} was not found: ${filename}`);
    process.exit(1);
  }
}

function requireDirectory(directory, label) {
  if (!fs.existsSync(directory) || !fs.statSync(directory).isDirectory()) {
    console.error(`${label} was not found: ${directory}`);
    process.exit(1);
  }
}

function smokeTest(executable, flag, label) {
  console.log(`\n> Smoke-testing Nuitka sidecar (${label})`);
  const result = spawnSync(executable, [flag], {
    cwd: ROOT,
    encoding: "utf-8",
    windowsHide: true,
  });
  const output = `${result.stdout || ""}\n${result.stderr || ""}`;
  let diagnostic = null;
  try {
    diagnostic = JSON.parse((result.stdout || "").trim());
  } catch {}
  if (result.error || result.status !== 0 || diagnostic?.available !== true) {
    console.error(result.error?.message || output.slice(-2000));
    console.error(`Nuitka sidecar ${label} smoke test failed; refusing to stage it.`);
    process.exit(1);
  }
  console.log(`  Sidecar ${label} verified`);
}

function main() {
  console.log("\n============================================================");
  console.log("  ASMRTranslator — Nuitka Python sidecar build");
  console.log(`  Build profile: ${FAST_BUILD ? "fast CI" : "release LTO"}`);
  console.log("============================================================");

  requireFile(PACKAGE_CONFIG, "Nuitka package configuration");
  requireFile(GGUF, "FSMN-VAD model");
  requireFile(VAD_DLL, "FSMN-VAD runtime DLL");

  run("uv", ["sync", "--all-groups", "--frozen"]);

  requireDirectory(UNIDIC_DIR, "UniDic Lite dictionary");
  for (const filename of ["dicrc", "sys.dic", "matrix.bin", "char.bin", "unk.dic"]) {
    requireFile(path.join(UNIDIC_DIR, filename), `UniDic Lite ${filename}`);
  }

  removeDirectory(OUTPUT_DIR);
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.mkdirSync(RESOURCE_DIR, { recursive: true });
  const logDefaults = path.join(OUTPUT_DIR, "_log_upload_defaults.json");
  run("uv", ["run", "--frozen", "python", "scripts/prepare_log_defaults.py",
    "--output", logDefaults]);

  const nuitkaArgs = [
    "run",
    "python",
    "-m",
    "nuitka",
    path.join(ROOT, "main.py"),
    "--mode=onefile",
    "--output-filename=main.exe",
    `--output-dir=${OUTPUT_DIR}`,
    "--assume-yes-for-downloads",
    "--deployment",
    "--python-flag=-OO",
    // Release builds favor the smallest optimized binary on memory-constrained
    // local machines. CI fast builds trade LTO for four-way C compilation;
    // runtime contracts are still checked by both post-build smoke tests.
    ...(FAST_BUILD ? ["--jobs=4", "--lto=no"] : ["--low-memory", "--lto=yes"]),
    // Keep development-only frameworks out of the standalone import graph.
    // Nuitka still follows every runtime import used by the application.
    "--noinclude-setuptools-mode=nofollow",
    "--noinclude-pytest-mode=nofollow",
    "--noinclude-unittest-mode=nofollow",
    "--noinclude-pydoc-mode=nofollow",
    "--noinclude-IPython-mode=nofollow",
    "--noinclude-dask-mode=nofollow",
    "--noinclude-numba-mode=nofollow",
    "--include-package-data=pykakasi",
    "--include-package-data=certifi",
    "--include-package-data=jsonschema_specifications",
    "--include-package=socksio",
    "--include-package=aiohttp_socks",
    "--include-package=python_socks",
    `--include-data-dir=${UNIDIC_DIR}=unidic_lite/dicdir`,
    // Nuitka classifies .bin files as binary modules and excludes them from a
    // generic data directory. These two files are MeCab dictionary data, not
    // executables, so force them into the onefile payload as data files.
    `--include-data-files=${path.join(UNIDIC_DIR, "char.bin")}=unidic_lite/dicdir/char.bin`,
    `--include-data-files=${path.join(UNIDIC_DIR, "matrix.bin")}=unidic_lite/dicdir/matrix.bin`,
    `--include-data-files=${GGUF}=modules/vad/model/fsmn-vad.gguf`,
    `--include-data-files=${logDefaults}=modules/_log_upload_defaults.json`,
    `--user-package-configuration-file=${PACKAGE_CONFIG}`,
    `--report=${path.join(OUTPUT_DIR, "compilation-report.xml")}`,
  ];

  run("uv", nuitkaArgs);

  requireFile(BUILT_EXE, "Nuitka output");
  smokeTest(BUILT_EXE, "--check-vad-runtime", "VAD runtime");
  smokeTest(BUILT_EXE, "--check-japanese-runtime", "MeCab/UniDic runtime");
  fs.copyFileSync(BUILT_EXE, SIDECAR);

  const sizeMB = (fs.statSync(SIDECAR).size / 1024 / 1024).toFixed(1);
  console.log(`\n  Python sidecar ready: ${SIDECAR} (${sizeMB} MB)`);

  removeDirectory(OUTPUT_DIR);
  console.log("  Cleaned Nuitka intermediate output");
}

main();
