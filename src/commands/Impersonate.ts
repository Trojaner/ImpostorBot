import {SlashCommandBuilder} from '@discordjs/builders';
import {ChannelType} from 'discord.js';
import {Command} from '../Command';
import {generateTextFromDiscordMessages} from '../rnn/DiscordTextGenerator';
import Messages from '../Messages';

export default new Command({
  private: true,

  builder: new SlashCommandBuilder()
    .addUserOption(option =>
      option
        .setName('user')
        .setDescription('User to impersonate.')
        .setRequired(true)
    )
    .addChannelOption(option =>
      option
        .setName('channel')
        .setDescription('Channel to post to.')
        .setRequired(true)
    )
    .addStringOption(option =>
      option
        .setName('prompt')
        .setDescription('The optional prompt.')
        .setRequired(false)
    )
    .setName('impersonate')
    .setDescription('Impersonate a user'),

  run: async ({interaction, client}) => {
    await interaction.deleteReply();

    const userId = interaction.options.get('user')!.value as string;

    const channelId = interaction.options.get('channel')!.value as string;
    const channel = await client.channels.fetch(channelId);

    if (channel == null) {
      await interaction.followUp({
        embeds: [Messages.error().setDescription('Channel not found.')],
      });

      return;
    }

    if (channel.type != ChannelType.GuildText) {
      await interaction.followUp({
        embeds: [
          Messages.error().setDescription('Channel is not a text channel.'),
        ],
      });

      return;
    }

    const query = interaction.options.get('query')?.value as string;
    await generateTextFromDiscordMessages({
      client,
      channel,
      userId,
      query,
      interaction,
    });
  },
});
