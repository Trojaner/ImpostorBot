import {Event} from '../Event';

export default new Event({
  name: 'ready',
  run: async client => {
    for (const guild of client.guilds.cache.values()) {
      await client.deployCommands(guild);
    }
  },
});
