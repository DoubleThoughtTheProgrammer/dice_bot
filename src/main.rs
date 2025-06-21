use std::env;

use serenity::all::{Command, CreateInteractionResponse, CreateInteractionResponseMessage, GuildId, Interaction, Ready};
use serenity::{all::EventHandler, async_trait};
use serenity::prelude::*;
mod commands;
mod parser;

struct Handler;

#[async_trait]
impl EventHandler for Handler {
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        if let Interaction::Command(command) = interaction {
            let response = match command.data.name.as_str() {
                "roll" => Some(commands::roll::run(&command.data.options)),
                _ => Some("not implemented".to_string()),
            };

            if let Some(content) = response {
                let data = CreateInteractionResponseMessage::new().ephemeral(false).content(content);
                let builder = CreateInteractionResponse::Message(data);
                if let Err(why) = command.create_response(&ctx.http, builder).await {
                    println!("Cannot respond to slash command: {why}");
                }
            }
        }
    }

    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} connected", ready.user.name);

        let test_guild = env::var("DISCORD_TEST_GUILD_ID");
        if let Ok(test_guild_id) = test_guild {
            let test_guild_id = GuildId::new(test_guild_id.parse().expect("Invalid DISCORD_TEST_GUILD_ID: Not an integer"));

            let test_commands = test_guild_id.set_commands(&ctx.http, vec![
                // commands::roll::register()
            ]).await;

            if let Err(why) = test_commands {
                println!("Failed to create test commands: {why}");
            }
        }

        let global_command = Command::create_global_command(&ctx.http, commands::roll::register()).await;
        if let Err(why) = global_command {
            println!("Failed to create global command: {why}");
        }
    }
}

#[tokio::main]
async fn main() {
    // Configure the client with your Discord bot token in the environment.
    let token = env::var("DISCORD_TOKEN").expect("Expected a token in the environment");
    // Set gateway intents, which decides what events the bot will be notified about
    let intents = GatewayIntents::empty();
    // Create a new instance of the Client, logging in as a bot. This will automatically prepend
    // your bot token with "Bot ", which is a requirement by Discord for bot users.
    let mut client =
        Client::builder(&token, intents).event_handler(Handler).await.expect("Err creating client");

    // Finally, start a single shard, and start listening to events.
    //
    // Shards will automatically attempt to reconnect, and will perform exponential backoff until
    // it reconnects.
    if let Err(why) = client.start().await {
        println!("Client error: {why:?}");
    }
}
