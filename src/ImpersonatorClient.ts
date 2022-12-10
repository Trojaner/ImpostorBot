import path from 'path';
import requireAll from 'require-all';
import {Command} from './Command';
import {Event} from './Event';
import {handleEvent} from './Handlers';
import {Client, Collection, Guild} from 'discord.js';

export default class ImpersonatorClient extends Client {
  commands: Collection<string, Command> = new Collection();

  constructor() {
    super({
      intents: ['GuildMessages'],
    });
  }

  async start() {
    await this.resolveModules();
    await this.login(process.env.DISCORD_CLIENT_TOKEN);
  }

  async resolveModules() {
    const sharedSettings = {
      recursive: true,
      filter: /\w*.[tj]s/g,
    };

    requireAll({
      ...sharedSettings,
      dirname: path.join(__dirname, './commands'),
      resolve: x => {
        const command = x.default as Command;

        if (command.disabled) return;

        console.log(`Command '${command.builder.name}' registered.`);
        this.commands.set(command.builder.name, command);
      },
    });

    requireAll({
      ...sharedSettings,
      dirname: path.join(__dirname, './events'),
      resolve: x => {
        const event = x.default as Event;
        console.log(`Event '${event.name}' registered.`);
        handleEvent(this, event);
      },
    });
  }

  async deployCommands(guild: Guild) {
    const commandsJSON = [...this.commands.values()].map(x =>
      x.builder.toJSON()
    );

    await guild.commands.set(commandsJSON);
  }
}
