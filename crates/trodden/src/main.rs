mod cli;
mod hook;
mod mcp;
mod output;

use std::{env, process::ExitCode};

fn main() -> ExitCode {
    let mut args = env::args_os().skip(1);
    if args.next().is_some_and(|command| command == "hook") {
        return hook::Hook::run(args.next().as_deref());
    }
    cli::Cli::run()
}
