use assert_cmd::Command;
use predicates::str::contains;

const BIN: &str = "task";

#[test]
fn displays_help() {
    Command::cargo_bin(BIN)
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(contains("EXAMPLES:"));
}

#[test]
fn warns_on_empty_description() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["gen", ""])
        .env(
            "TASK_SH_FAKE_RESPONSE",
            "Command: echo noop\nExplanation: noop",
        )
        .assert()
        .success()
        .stdout(contains("Please provide more details"));
}

#[test]
fn accepts_stdin_description() {
    Command::cargo_bin(BIN)
        .unwrap()
        .arg("gen")
        .env(
            "TASK_SH_FAKE_RESPONSE",
            "Command: du -sh *\nExplanation: summarize disk usage",
        )
        .write_stdin("check disk usage")
        .assert()
        .success()
        .stdout(contains("Suggested command"));
}

#[test]
fn verbose_mode_shows_raw_response() {
    Command::cargo_bin(BIN)
        .unwrap()
        .args(["gen", "list files", "--verbose"])
        .env(
            "TASK_SH_FAKE_RESPONSE",
            "Command: ls -la\nExplanation: Lists files",
        )
        .assert()
        .success()
        .stdout(contains("Raw response:"))
        .stdout(contains("Explanation:"));
}
