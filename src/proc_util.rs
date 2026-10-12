use std::ffi::OsStr;
use std::process::Command;

/// Strips characters that make an otherwise-correct path fail to launch:
/// surrounding whitespace, and a matching pair of surrounding quotes
/// (straight `"`/`'` or the "smart" quotes Word/some editors substitute).
/// `Command::new` / Windows' `CreateProcess` take the string completely
/// literally — no shell in between to strip quotes for us — so a path
/// pasted from Windows Explorer's "Copier en tant que chemin d'accès"
/// (which wraps it in `"..."`) makes Windows look for a program whose name
/// literally starts and ends with `"`, which doesn't exist. That fails with
/// `os error 123` (ERROR_INVALID_NAME, "La syntaxe du nom de fichier... est
/// incorrecte"), which otherwise looks exactly like a wrong/missing path.
fn sanitize_program_path(s: &str) -> String {
    let t = s.trim();
    let t = t
        .strip_prefix(['"', '\'', '\u{201C}', '\u{2018}'])
        .unwrap_or(t);
    let t = t
        .strip_suffix(['"', '\'', '\u{201D}', '\u{2019}'])
        .unwrap_or(t);
    t.trim().to_string()
}

/// Builds a `Command` for `program` with `args`, optionally hinting the OS
/// scheduler to run it at reduced priority — so a long batch (whisper-cli,
/// Python/XTTS) doesn't make the machine feel frozen for other work.
///
/// Windows: uses `CREATE_NO_WINDOW`-free `BELOW_NORMAL_PRIORITY_CLASS` via
/// the stdlib's `creation_flags` (no extra crate needed).
/// Linux/macOS: wraps the command with `nice -n 10`.
///
/// This is a best-effort scheduling hint, not a hard CPU cap: the OS still
/// gives the process CPU time when nothing else wants it, it just yields
/// faster to foreground/interactive work.
///
/// `program` is sanitized (see `sanitize_program_path`) before launch, as a
/// last line of defense even if a stray-quoted path slipped into Settings
/// (typed/pasted by hand, or loaded from an older saved session).
pub fn command_with_priority<S: AsRef<OsStr>>(
    program: &str,
    args: &[S],
    low_priority: bool,
) -> Command {
    let program = sanitize_program_path(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        let mut cmd = Command::new(&program);
        cmd.args(args);
        if low_priority {
            cmd.creation_flags(BELOW_NORMAL_PRIORITY_CLASS);
        }
        cmd
    }
    #[cfg(not(windows))]
    {
        let mut cmd = if low_priority {
            let mut c = Command::new("nice");
            c.arg("-n").arg("10").arg(&program);
            c
        } else {
            Command::new(&program)
        };
        cmd.args(args);
        cmd
    }
}
