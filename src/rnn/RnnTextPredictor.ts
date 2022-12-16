import * as tf from '@tensorflow/tfjs-node';
import {io as tfio} from '@tensorflow/tfjs';

export type ExportedModel = {
  modelTopology: any;
  weightSpecs: tfio.WeightsManifestEntry[];
  weightData: ArrayBuffer;
  tokenizedData: TokenizedData;
};

export type TokenizedData = {
  vocabulary: string[];
  charToIndex: {[char: string]: number};
  indexToChar: {[index: number]: string};
};

export default class RnnTextPredictor {
  model: tf.LayersModel | null = null;
  tokenizedData: TokenizedData | null = null;

  constructor() {}

  async train(data: string[]) {
    // Preprocess data
    const vocabulary = [...new Set(data.map(x => x + '\n').join(''))];
    const charToIndex = vocabulary.reduce(
      (obj, char, i) => ({...obj, [char]: i}),
      {}
    );

    const indexToChar = vocabulary.reduce(
      (obj, char, i) => ({...obj, [i]: char}),
      {}
    );

    this.tokenizedData = {
      vocabulary,
      charToIndex,
      indexToChar,
    };

    // Convert data to one-hot encoded tensors
    const xs = data.map(str => this.strToXs(str));
    const ys = data.map(str => this.strToYs(str));
    const xsTensor = tf.stack(xs);
    const ysTensor = tf.stack(ys);

    // Build model
    const inputShape = [1, this.tokenizedData.vocabulary.length];
    this.model = tf.sequential({
      layers: [
        tf.layers.lstm({units: 64, inputShape}),
        tf.layers.dense({
          units: this.tokenizedData.vocabulary.length,
          activation: 'softmax',
        }),
      ],
    });

    this.model.compile({loss: 'categoricalCrossentropy', optimizer: 'adamax'});

    // Train model
    await this.model.fit(xsTensor, ysTensor, {epochs: 10});
  }

  predictRemainder(text: string): string {
    if (!this.model?.built || !this.tokenizedData)
      throw new Error('Model not trained yet.');

    const xs = this.strToXs(text);
    let xsTensor = tf.tensor2d(xs, [1, this.tokenizedData.vocabulary.length]);

    let result = '';
    let y: tf.Tensor;
    while (result.slice(-text.length) !== text) {
      y = this.model.predict(xsTensor) as tf.Tensor;
      const index = y.argMax(-1).dataSync()[0];
      result += this.tokenizedData.indexToChar[index];
      xsTensor.dispose();
      xsTensor = y.expandDims(0);
    }

    return result.slice(text.length);
  }

  predictRelated(text: string): string {
    if (!this.model?.built || !this.tokenizedData)
      throw new Error('Model not trained yet.');

    const xs = this.strToXs(text);
    const xsTensor = tf.tensor2d(xs, [1, this.tokenizedData.vocabulary.length]);

    let result = '';
    let y: tf.Tensor;
    while (result.length < text.length) {
      y = this.model.predict(xsTensor) as tf.Tensor;
      const index = y.argMax(-1).dataSync()[0];
      result += this.tokenizedData.indexToChar[index];
      xsTensor.dispose();
    }

    return result;
  }

  generateRandom(): string {
    if (!this.model?.built || !this.tokenizedData)
      throw new Error('Model not trained yet.');

    let result = '';
    let xsTensor = tf.tensor2d(this.strToXs(this.tokenizedData.vocabulary[0]), [
      1,
      this.tokenizedData.vocabulary.length,
    ]);

    let y: tf.Tensor;
    while (true) {
      y = this.model.predict(xsTensor) as tf.Tensor;
      const index = y.argMax(-1).dataSync()[0];
      const char = this.tokenizedData.indexToChar[index];
      if (char === '\n') break;
      result += char;
      xsTensor.dispose();
      xsTensor = y.expandDims(0);
    }

    return result;
  }

  async export(): Promise<ExportedModel> {
    if (!this.model?.built || !this.tokenizedData)
      throw new Error('Model not trained yet.');

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
      tokenizedData: this.tokenizedData,
    };
  }

  async import(model: ExportedModel): Promise<void> {
    const modelIo = tf.io.fromMemory(model);
    this.model = await tf.loadLayersModel(modelIo);
    this.tokenizedData = model.tokenizedData;
  }

  dispose() {
    if (this.model?.built) {
      this.model?.dispose();
    }
  }

  private strToXs(str: string): number[][] {
    if (!this.tokenizedData) throw new Error('Tokenized data not available.');

    return str.split('').map(char => {
      const x = new Array(this.tokenizedData!.vocabulary.length).fill(0);
      x[this.tokenizedData!.charToIndex[char]] = 1;
      return x;
    });
  }

  private strToYs(str: string): number[][] {
    if (!this.tokenizedData) throw new Error('Tokenized data not available.');

    return str
      .slice(1)
      .split('')
      .map(char => {
        const y = new Array(this.tokenizedData!.vocabulary.length).fill(0);
        y[this.tokenizedData!.charToIndex[char]] = 1;
        return y;
      });
  }
}
