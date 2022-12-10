import {SlashCommandBuilder} from '@discordjs/builders';
import {ChannelType, Snowflake} from 'discord.js';
import {Op} from 'sequelize';
import {Command} from '../Command';
import {DbMessages} from '../Database';
import Messages from '../Messages';

export default new Command({
  builder: new SlashCommandBuilder()
    .addChannelOption(option =>
      option
        .setName('channel')
        .setDescription('Channel to import from.')
        .setRequired(true)
    )
    .addUserOption(option =>
      option
        .setName('user')
        .setDescription('User to import from.')
        .setRequired(false)
    )

    .setName('import')
    .setDescription('Import messages to database.'),

  run: async ({interaction, client}) => {
    const channelId = interaction.options.get('channel')!.value as string;
    const channel = await client.channels.fetch(channelId);

    if (channel == null) {
      await interaction.editReply({
        embeds: [Messages.error().setDescription('Channel not found.')],
      });

      return;
    }

    if (channel.type != ChannelType.GuildText) {
      await interaction.editReply({
        embeds: [
          Messages.error().setDescription('Channel is not a text channel.'),
        ],
      });

      return;
    }

    const userId =
      (interaction.options.get('user')?.value as string) || undefined;

    const latestMessage = await DbMessages.findOne({
      where: {
        channel_id: channelId,
        user_id: userId || {[Op.ne]: null},
      },
      order: [['time', 'ASC']],
    });

    let latestMessageId: Snowflake | undefined = undefined;
    let count = await DbMessages.count({
      where: {
        channel_id: channelId,
        user_id: userId || {[Op.ne]: null},
      },
    });

    if (latestMessage) {
      latestMessageId = latestMessage.getDataValue<Snowflake>('message_id');

      await interaction.followUp({
        embeds: [
          Messages.success().setDescription(
            `Continuing to import after message # ${latestMessageId}`
          ),
        ],
      });
    }

    while (true) {
      let messages = await channel.messages.fetch({
        before: latestMessageId,
        cache: false,
        limit: 100,
      });

      if (messages.size == 0) {
        break;
      }

      messages = messages.filter(
        message => !message.author.bot && !message.author.system
      );

      await interaction.editReply({
        embeds: [
          Messages.success().setDescription('Imported messages: ' + count),
        ],
      });

      await DbMessages.bulkCreate(
        messages.map(message => ({
          guild_id: message.guild.id,
          message_id: message.id,
          user_id: message.author.id,
          channel_id: message.channel.id,
          content: message.content,
          time: message.createdAt,
        }))
      );

      count += messages.size;
      latestMessageId = messages.last()?.id;
    }

    await interaction.followUp({
      embeds: [
        Messages.success().setDescription(`Done: Imported ${count} messages`),
      ],
    });
  },
});
