import {
  CommandInteraction,
  GuildMember,
  Snowflake,
  TextChannel,
  Webhook,
  WebhookClient,
} from 'discord.js';
import {Model, Op} from 'sequelize';
import {setTimeout} from 'timers/promises';
import {DbMessages, DbModels} from '../Database';
import ImpersonatorClient from '../ImpersonatorClient';
import Messages from '../Messages';
import RnnTextPredictor, {ExportedModel} from './RnnTextPredictor';

interface IGenerateTextOptions {
  client: ImpersonatorClient;
  interaction?: CommandInteraction;
  query?: string;
  userId?: Snowflake;
  guildId?: Snowflake;
  channel: TextChannel;
}

export async function generateTextFromDiscordMessages(
  options: IGenerateTextOptions
) {
  const result = await predictText(options);

  if (!result) {
    await options.interaction?.followUp({
      embeds: [
        Messages.error().setDescription('Failed to generate a message.'),
      ],
      ephemeral: true,
    });

    return;
  }

  const {client, interaction, channel, userId} = options;

  let webhook: Webhook | undefined;

  try {
    const webhooks = await channel.fetchWebhooks();
    webhook = webhooks.find(x => x.name == 'Impersonator Bot');

    if (webhook == null) {
      webhook = await channel.createWebhook({
        name: 'Impersonator Bot',
        reason: 'Impersonator bot user impersonation',
      });
    }
  } catch (e: any) {
    await interaction?.followUp({
      embeds: [
        Messages.error().setDescription(
          'Failed to send webhook. Please make sure I have the `Manage Webhooks` permission.'
        ),
      ],
      ephemeral: true,
    });

    console.log(e);
    return;
  }

  const webhookClient = new WebhookClient({url: webhook.url});

  let member: GuildMember;
  if (userId) {
    member = await channel.guild.members.fetch(userId);
  } else {
    member = await channel.guild.members.fetch(client.user!.id);
  }

  try {
    await webhookClient.send({
      content: result,
      avatarURL: member.displayAvatarURL() || member.avatarURL() || undefined,
      username: member.displayName || member.nickname || member.user.username,
    });
  } catch (e: any) {
    if (e.toString().includes('USERNAME_INVALID')) {
      await webhookClient.send({
        content: result,
        avatarURL: member.displayAvatarURL() || member.avatarURL() || undefined,
        username: member.user.username,
      });
    }
  }
}

async function predictText(
  options: IGenerateTextOptions
): Promise<string | null> {
  const {channel, interaction, query, guildId, userId} = options;

  let input: string | null = null;
  if (query && query.length > 0) {
    input = query || null;
  }

  let dbModel = await DbModels.findOne({
    where: {
      guild_id: guildId || channel.guild.id,
      user_id: userId || {[Op.ne]: null},
    },
    order: [['id', 'DESC']],
  });

  const lastKnownMessage = await DbMessages.findOne({
    where: {
      guild_id: guildId || channel.guild.id,
      user_id: userId || {[Op.ne]: null},
    },
    order: [['time', 'DESC']],
  });

  if (!lastKnownMessage) {
    await interaction?.followUp({
      embeds: [
        Messages.error().setDescription(
          'Failed to generate a message. No messages found for this user.'
        ),
      ],
      ephemeral: true,
    });

    console.log('No last known message found');
    return null;
  }

  const lastKnownMessageId =
    lastKnownMessage.getDataValue<Snowflake>('message_id');

  let oldModel: Model | any = null;

  if (
    dbModel &&
    dbModel.getDataValue<Snowflake>('last_message_id') != lastKnownMessageId
  ) {
    const count = await DbMessages.count({
      where: {
        guild_id: guildId || channel.guild.id,
        user_id: userId || {[Op.ne]: null},
        time: {
          [Op.gt]: dbModel.getDataValue('update_time'),
        },
      },
    });

    if (count > 5000) {
      // model is outdated, delete it
      oldModel = dbModel;
      dbModel = null;
    }
  }

  let model: ExportedModel | null = null;
  if (dbModel) {
    const buffer = dbModel.getDataValue('weight_data') as Buffer;
    model = {
      modelTopology: JSON.parse(dbModel.getDataValue<string>('model_topology')),
      weightSpecs: JSON.parse(dbModel.getDataValue<string>('weight_specs')),
      weightData: buffer,
      tokenizedData: JSON.parse(dbModel.getDataValue<string>('tokenized_data')),
    };
  }

  const rnn = new RnnTextPredictor();
  if (model) {
    await rnn.import(model);
  } else {
    const entries = await DbMessages.findAll({
      where: {
        user_id: userId || {[Op.ne]: null},
        guild_id: guildId || channel.guild.id,
        content: {[Op.ne]: ''},
      },
    });

    const commandPrefixes = ['!', '?', '-', '+', '@', '#', '$', '%', '&'];

    const messages = entries
      .map(x => x.getDataValue<string>('content'))
      .filter(x => x != '' && x)
      .filter(
        x =>
          !commandPrefixes.some(prefix => x.startsWith(prefix) && x.length > 2)
      );

    if (messages.length == 0) {
      await interaction?.followUp({
        embeds: [
          Messages.error().setDescription(
            'Failed to generate a message. No messages found for this user.'
          ),
        ],
        ephemeral: true,
      });

      console.error('No messages found');
      return null;
    }

    const message = await interaction?.followUp({
      embeds: [
        Messages.error().setDescription(
          'User model is not trained or outdated. Training model now, this can take a while...'
        ),
      ],
      ephemeral: true,
    });

    await setTimeout(1000);

    console.log(`Train started with ${messages.length} messages`);
    await rnn.train(messages, {
      onBatchBegin: async (batch, logs) => {
        await message?.edit({
          embeds: [
            Messages.error().setDescription(
              `Training model... Batch ${batch} of ${logs.batches}`
            ),
          ],
        });
      },
    });

    console.log('Train finished, saving model...');

    const {modelTopology, weightSpecs, weightData, tokenizedData} =
      await rnn.export();

    await DbModels.create({
      guild_id: guildId || channel.guild.id,
      user_id: userId || null,
      model_topology: JSON.stringify(modelTopology),
      weight_specs: JSON.stringify(weightSpecs),
      weight_data: weightData,
      tokenized_data: JSON.stringify(tokenizedData),
      last_message_id: lastKnownMessageId,
    });

    if (oldModel) {
      await oldModel.destroy();
    }

    console.log('Model saved to db');
    await message?.edit(
      'Model trained and saved to database. Generating message...'
    );
  }

  if (input) {
    return rnn.predictRemainder(input!);
  }

  return rnn.generateRandom();
}
