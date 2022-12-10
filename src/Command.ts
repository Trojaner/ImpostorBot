import {CommandInteraction} from 'discord.js';
import {SlashCommandBuilder} from '@discordjs/builders';

import ImpersonatorClient from './ImpersonatorClient';

export type CommandArgs = {
  client: ImpersonatorClient;
  interaction: CommandInteraction;
};

export class Command {
  disabled?: boolean;

  private?: boolean;

  builder!: SlashCommandBuilder;

  run!: (args: CommandArgs) => any;

  constructor(options: NonNullable<Command>) {
    Object.assign(this, options);
  }
}
