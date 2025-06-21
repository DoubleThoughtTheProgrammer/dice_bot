pub(crate) mod roll {
    use serenity::all::{CommandDataOption, CommandOptionType, CreateCommand, CreateCommandOption};

    use crate::parser::{do_dice_command, Error};

    pub(crate) fn run(options: &Vec<CommandDataOption>) -> String {
        let command = options[0].value.as_str().expect("Should always be a string param");
        let command_result = do_dice_command(&command);

        match command_result {
            Ok(num) => {
                ["Rolling ".to_string(), command.to_string(), ":\n".to_string(), num.to_string()].concat()
            },
            Err(Error {position: index, error_type: _ }) => {
                let error_here = "error here";
                let len_eh = error_here.len();
                let result_len: usize = 4 + command.len() + 1 + index + if index > len_eh {0} else {len_eh} + 3;
                let mut result = String::with_capacity(result_len);
                result.push_str("```\n");
                result.push_str(command);
                result.push('\n');
                if index < len_eh {
                    for _ in 0..index {
                        result.push(' ');
                    }
                    result.push('^');
                    result.push_str(error_here);
                } else {
                    for _ in 0..(index - len_eh) {
                        result.push(' ');
                    }
                    result.push_str(error_here);
                    result.push('^');
                }
                result.push_str("```");
                result
            }
        }
    }

    pub(crate) fn register() -> CreateCommand {
        CreateCommand::new("roll")
            .description("Do a roll command")
            .add_option(
                CreateCommandOption::new(
                    CommandOptionType::String, "command", "The format of the roll"
                ).required(true)
            )
    }
}