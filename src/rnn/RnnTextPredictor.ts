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
  wordToIndex: {[word: string]: number};
  indexToWord: {[index: number]: string};
};

export default class RnnTextPredictor {
  model: tf.LayersModel | null = null;
  tokenizedData: TokenizedData | null = null;

  constructor() {}

  async train(data: string[]) {
    const maxVocubularySize = 1024;

    // Preprocess data
    data = data.sort(() => Math.random() - 0.5).slice(0, 5000);

    const wordCounts: {[word: string]: number} = {};
    data.forEach(str => {
      str.split(' ').forEach(word => {
        if (!wordCounts[word]) {
          wordCounts[word] = 0;
        }
        wordCounts[word]++;
      });
    });

    const sortedWords = Object.keys(wordCounts).sort(
      (a, b) => wordCounts[b] - wordCounts[a]
    );
    const vocabulary = sortedWords.slice(0, maxVocubularySize);

    const wordToIndex = vocabulary.reduce(
      (obj, word, i) => ({...obj, [word]: i}),
      {}
    );
    const indexToWord = vocabulary.reduce(
      (obj, word, i) => ({...obj, [i]: word}),
      {}
    );

    this.tokenizedData = {
      vocabulary,
      wordToIndex,
      indexToWord,
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
        tf.layers.lstm({units: 16, inputShape, batchSize: 8}),
        tf.layers.dense({
          units: this.tokenizedData.vocabulary.length,
          batchSize: 8,
          activation: 'softmax',
        }),
      ],
    });

    this.model.compile({loss: 'categoricalCrossentropy', optimizer: 'adamax'});

    // Train model
    await this.model.fit(xsTensor, ysTensor, {epochs: 5});
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
      result += this.tokenizedData.indexToWord[index] + ' ';
      xsTensor.dispose();
      xsTensor = y.expandDims(0);
    }

    return result.slice(text.length);
  }

  generateRandom(): string {
    if (!this.model?.built || !this.tokenizedData)
      throw new Error('Model not trained yet.');

    const randomWord =
      this.tokenizedData.vocabulary[
        Math.floor(Math.random() * this.tokenizedData.vocabulary.length)
      ];

    return this.predictRemainder(randomWord);
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
    if (!this.tokenizedData) throw new Error('Tokenized data not set yet');

    return this.pad(str.split(' ')).map(word => {
      const wordIndex = this.tokenizedData!.wordToIndex[word] || 0;
      const x = new Array(this.tokenizedData!.vocabulary.length).fill(0);
      x[wordIndex] = 1;
      return x;
    });
  }

  private strToYs(str: string): number[][] {
    if (!this.tokenizedData) throw new Error('Tokenized data not set yet');

    return this.pad(str.split(' ').slice(1)).map(word => {
      const wordIndex = this.tokenizedData!.wordToIndex[word] || 0;
      const y = new Array(this.tokenizedData!.vocabulary.length).fill(0);
      y[wordIndex] = 1;
      return y;
    });
  }

  private pad(str: string[]): string[] {
    if (!this.tokenizedData) throw new Error('Tokenized data not set yet');

    const length = Math.max(
      ...this.tokenizedData?.vocabulary.map(s => s.split(' ').length)
    );

    while (str.length < length) {
      str.push('');
    }

    return str;
  }
}
