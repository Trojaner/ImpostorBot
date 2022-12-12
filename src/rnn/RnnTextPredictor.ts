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
    const tokenizer = new tf_tokenizer.Tokenizer();
    tokenizer.fitOnTexts(data);
    const encodedData = tokenizer.textsToSequences(data);
    const paddedData = tf_tokenizer.padSequences(encodedData, {maxlen: 50});
    const inputs = tf.tensor2d(paddedData, [
      paddedData.length,
      paddedData[0].length,
    ]);
    const labels = tf.ones([paddedData.length, 1]);

    this.model = tf.sequential({
      layers: [
        tf.layers.bidirectional({
          layer: tf.layers.lstm({units: 128}),
          mergeMode: 'concat',
          inputShape: [encodedData.input.length],
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
    const paddedInput = tf_tokenizer.padSequences(encodedInput, {maxlen: 50});
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
