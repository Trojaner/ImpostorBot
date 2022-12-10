import {Guild} from 'discord.js';
import {Event} from '../Event';

export default new Event({
  name: 'guildCreate',
  run: async (client, guild: Guild) => {
    await client.deployCommands(guild);
  },
});
