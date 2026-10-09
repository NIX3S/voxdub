use std::ffi::OsStr;
use std::process::Command;

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
pub fn command_with_priority<S: AsRef<OsStr>>(
    program: &str,
    args: &[S],
    low_priority: bool,
) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;
        let mut cmd = Command::new(program);
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
            c.arg("-n").arg("10").arg(program);
            c
        } else {
            Command::new(program)
        };
        cmd.args(args);
        cmd
    }
}
