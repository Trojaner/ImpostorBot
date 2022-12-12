import * as tf from '@tensorflow/tfjs-node';
import {io as tfio} from '@tensorflow/tfjs';
import * as tf_tokenizer from 'tf_node_tokenizer';

export type TextMessage = string;

export type ExportedModel = {
  modelTopology: any;
  weightSpecs: tfio.WeightsManifestEntry[];
  weightData: ArrayBuffer;
};

export default class RnnTextPredictor {
  private model: tf.LayersModel | null = null;

  async train(data: TextMessage[]) {
    const tokenizer = new tf_tokenizer.Tokenizer({
      oov_token: '<unk>',
      lower: false,
    });

    tokenizer.fitOnTexts(data);

    const encodedData = tokenizer.textsToSequences(
      data.filter(x => x && x.trim() != '')
    );

    const padLength = 512;
    const paddedData = this.padSequences(encodedData, {maxlen: padLength});

    const inputs = tf.tensor2d(paddedData, [paddedData.length, padLength]);

    const labels = tf.ones([paddedData.length, 1]);

    this.model = tf.sequential({
      layers: [
        tf.layers.bidirectional({
          layer: tf.layers.lstm({units: 128}),
          mergeMode: 'concat',
          inputShape: [padLength, 1],
        }),
        tf.layers.dense({units: 128, activation: 'relu'}),
        tf.layers.dense({units: 1, activation: 'sigmoid'}),
      ],
    });

    this.model.compile({
      optimizer: tf.train.adamax(),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy'],
    });

    await this.model.fit(inputs, labels, {epochs: 5, batchSize: 32});
  }

  predictRemainder(input: TextMessage) {
    if (!this.model) throw new Error('Model not trained yet.');

    const tokenizer = new tf_tokenizer.Tokenizer();
    const encodedInput = tokenizer.textsToSequences(input);
    const paddedInput = this.padSequences(encodedInput, {maxlen: 255});
    const inputTensor = tf.tensor2d(paddedInput, [
      paddedInput.length,
      paddedInput[0].length,
    ]);

    const prediction = this.model.predict(inputTensor);

    const remainder = tokenizer.sequencesToTexts(prediction);
    return remainder;
  }

  generateRandom() {
    if (!this.model) throw new Error('Model not trained yet.');

    const tokenizer = new tf_tokenizer.Tokenizer();
    const prediction = this.model.predict(tf.randomNormal([1, 50]));
    const randomText = tokenizer.sequencesToTexts(prediction);
    return randomText;
  }

  padSequences(sequences: string[], {maxlen}) {
    const paddedData = sequences.map(sequence => {
      // If the length of the sequence is less than the maximum length,
      // pad the sequence with 0s to make it the same length as the maximum length.
      if (sequence.length < maxlen) {
        const padding = new Array(maxlen - sequence.length).fill(' ');
        return sequence.concat(padding.join(''));
      }
      // If the length of the sequence is equal to or greater than the maximum length,
      // truncate the sequence to the maximum length.
      return sequence.slice(0, maxlen);
    });

    return paddedData;
  }

  async export(): Promise<ExportedModel> {
    if (!this.model) throw new Error('Model not trained yet.');

    let modelArtifacts: tfio.ModelArtifacts | null = null;
    const modelTopologyType = 'JSON';

    const handler = tf.io.withSaveHandler(async artifacts => {
      modelArtifacts = artifacts;

      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType,
        },
        modelArtifacts,
      };
    });

    await this.model.save(handler, {
      includeOptimizer: true,
    });

    return {
      modelTopology: modelArtifacts!.modelTopology!,
      weightSpecs: modelArtifacts!.weightSpecs!,
      weightData: modelArtifacts!.weightData!,
    };
  }

  async import(model: ExportedModel): Promise<void> {
    const modelIo = tf.io.fromMemory(model);
    this.model = await tf.loadLayersModel(modelIo);
  }

  dispose() {
    if (this?.model?.built) {
      this.model?.dispose();
    }
  }
}
