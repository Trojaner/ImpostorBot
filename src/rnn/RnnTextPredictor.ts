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
  maxWordCount: number;
};

export default class RnnTextPredictor {
  model: tf.LayersModel | null = null;
  tokenizedData: TokenizedData | null = null;

  constructor() {}

  async train(data: string[]) {
    const maxMessages = 512;
    const maxVocubularySize = 768;
    const maxWordCount = 20;
    const batchSize = 8;
    const epochs = 5;

    // Preprocess data
    data = data
      .sort(() => Math.random() - 0.5)
      .map(msg => msg.replace(/\s{2,}/g, ' '))
      .filter(msg => msg.split(' ').length <= maxWordCount)
      .slice(0, maxMessages);

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
      maxWordCount,
    };

    // Convert data to one-hot encoded tensors
    const xs = data.map(str => this.strToXs(str));
    const ys = data.map(str => this.strToYs(str));

    const xsTensor = tf.stack(xs);
    let ysTensor = tf.stack(ys);
    ysTensor = ysTensor.reshape([
      batchSize,
      this.tokenizedData.vocabulary.length,
    ]);

    // Build model
    const inputShape = [maxWordCount, this.tokenizedData.vocabulary.length];
    this.model = tf.sequential({
      layers: [
        tf.layers.lstm({
          units: 16,
          inputShape,
          batchSize,
          returnSequences: true,
        }),
        tf.layers.lstm({units: 8, batchSize}),
        tf.layers.dense({
          units: this.tokenizedData.vocabulary.length,
          batchSize,
          activation: 'softmax',
        }),
      ],
    });

    this.model.compile({loss: 'categoricalCrossentropy', optimizer: 'adamax'});
    this.model.summary();

    // Train model
    await this.model.fit(xsTensor, ysTensor, {
      epochs,
      callbacks: {
        onBatchEnd: async (batch, logs) => {
          console.log(`Batch ${batch} / ${batchSize}: loss = ${logs?.loss}`);
        },
        onEpochEnd: async (epoch, logs) => {
          console.log(`Epoch ${epoch}/ ${epochs}: loss = ${logs?.loss}`);
        },
      },
    });

    xsTensor.dispose();
    ysTensor.dispose();
  }

  predictRemainder(text: string): string {
    if (!this.model?.built || !this.tokenizedData)
      throw new Error('Model not trained yet.');

    const xs = this.strToXs(text);
    let xsTensor = tf.tensor2d(xs, [1, this.tokenizedData.vocabulary.length]);

    let result = '';
    let y: tf.Tensor | null = null;
    while (result.slice(-text.length) !== text) {
      y = this.model.predict(xsTensor) as tf.Tensor;
      const index = y.argMax(-1).dataSync()[0];
      result += this.tokenizedData.indexToWord[index] + ' ';
      xsTensor.dispose();
      xsTensor = y.expandDims(0);
    }

    xsTensor.dispose();
    y?.dispose();

    return result.slice(text.length).replace(/\0/g, '');
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

    const padded = this.pad(str.split(' '));

    return padded.map(word => {
      const wordIndex = this.tokenizedData!.wordToIndex[word.trim()] || 0;

      const x = new Array(this.tokenizedData!.vocabulary.length).fill(0);
      if (!wordIndex) return x;

      x[wordIndex] = 1;
      return x;
    });
  }

  private strToYs(str: string): number[][] {
    if (!this.tokenizedData) throw new Error('Tokenized data not set yet');

    const padded = this.pad(str.split(' ').slice(1));

    return padded.map(word => {
      const wordIndex = this.tokenizedData!.wordToIndex[word.trim()] || 0;
      const y = new Array(this.tokenizedData!.vocabulary.length).fill(0);
      if (!wordIndex) return y;

      y[wordIndex] = 1;
      return y;
    });
  }

  private pad(str: string[]): string[] {
    if (!this.tokenizedData) throw new Error('Tokenized data not set yet');

    const length = this.tokenizedData.maxWordCount;

    if (str.length >= length) {
      return str.slice(0, length);
    }

    return [...str, ...Array(length - str.length).fill('\0')];
  }
}
