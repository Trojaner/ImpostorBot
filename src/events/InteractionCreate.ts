import {Interaction} from 'discord.js';

import {Event} from '../Event';
import {handleCommand} from '../Handlers';

export default new Event({
  name: 'interactionCreate',
  run: async (client, interaction: Interaction) => {
    if (interaction.isCommand()) {
      await handleCommand(client, interaction);
    }
  },
});
