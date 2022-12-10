import * as tf from '@tensorflow/tfjs-node';
import {io as tfio} from '@tensorflow/tfjs';
import natural from 'natural';

export type TextMessage = string;

export type ExportedModel = {
  modelTopology: any;
  weightSpecs: tfio.WeightsManifestEntry[];
  weightData: ArrayBuffer;
  tokenizedData: any;
};

export default class RnnTextPredictor {
  tokenizer: natural.WordTokenizer;
  model: tf.LayersModel | null = null;
  tokenizedData: any[] | null = null;

  constructor() {
    this.tokenizer = new natural.WordTokenizer();
  }

  async train(data: TextMessage[], callbacks?: any) {
    this.tokenizedData = data
      .filter(content => content && content.trim() != '')
      .map(content => this.tokenizer.tokenize(content));

    const encodedData = this.tokenizedData.map(sentence =>
      sentence.map(word => this.encodeWord(word))
    );

    const inputTensor = this.encodeBatch(encodedData);
    const outputTensor = this.encodeBatch(
      encodedData.map(sentence => sentence.slice(1))
    );

    this.model = this.createModel();
    this.model.compile({
      loss: 'categoricalCrossentropy',
      optimizer: 'adammax',
      metrics: ['accuracy'],
    });

    await this.model.fit(inputTensor, outputTensor, {
      epochs: 3,
      batchSize: 32,
      callbacks: {
        ...callbacks,
        onBatchBegin: async (batch, logs) => {
          console.log(`Batch ${batch}: loss = ${logs?.loss || 'N/A'}`);

          if (callbacks?.onBatchBegin) {
            callbacks?.onBatchBegin(batch, logs);
          }
        },
      },
    });
  }

  predictRemainder(input: string) {
    const tokenizedInput = this.tokenizer.tokenize(input);
    const encodedInput = tokenizedInput.map(word => this.encodeWord(word));
    const encodedRemainder = this.predictBatch(encodedInput);
    const decodedRemainder = encodedRemainder.map(wordIdx =>
      this.decodeWord(wordIdx)
    );
    return decodedRemainder.join(' ');
  }

  predictRelated(input: string) {
    const tokenizedInput = this.tokenizer.tokenize(input);
    const encodedInput = tokenizedInput.map(word => this.encodeWord(word));
    const encodedRemainder = this.predictBatch(encodedInput, true);
    const decodedRemainder = encodedRemainder.map(wordIdx =>
      this.decodeWord(wordIdx)
    );
    return decodedRemainder.join(' ');
  }

  generateRandom() {
    const encodedSentence = this.predictBatch([], true);
    const decodedSentence = encodedSentence.map(wordIdx =>
      this.decodeWord(wordIdx)
    );
    return decodedSentence.join(' ');
  }

  async export(): Promise<ExportedModel> {
    if (!this.model || !this.tokenizedData) {
      throw new Error(
        'Cannot export model: model not trained or imported yet.'
      );
    }

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
    this.tokenizedData = model.tokenizedData;

    const modelIo = tf.io.fromMemory(model);
    this.model = await tf.loadLayersModel(modelIo);
  }

  private createModel() {
    const model = tf.sequential();
    model.add(
      tf.layers.lstm({
        units: 32,
        returnSequences: true,
        recurrentInitializer: 'glorotNormal',
        inputShape: [null, this.getVocabularySize()],
      })
    );
    model.add(tf.layers.lstm({units: 16}));
    model.add(
      tf.layers.dense({units: this.getVocabularySize(), activation: 'softmax'})
    );
    return model;
  }

  private encodeBatch(data: number[][]) {
    const maxLen = Math.max(...data.map(sentence => sentence.length));
    const paddedData = data.map(sentence => {
      const padding = Array(maxLen - sentence.length).fill(0);
      return [...sentence, ...padding];
    });
    return tf.tensor2d(paddedData, [data.length, maxLen]);
  }

  private predictBatch(data: number[], random = false) {
    if (!this.model) {
      throw new Error(
        'Cannot predict batch: Model not trained or imported yet!'
      );
    }

    let encodedInput: number[];
    if (random) {
      encodedInput = Array(Math.floor(Math.random() * 10)).fill(0);
    } else {
      encodedInput = data.slice();
    }

    let output: number[] = [];
    for (let i = 0; i < 100; i++) {
      const inputTensor = this.encodeBatch([encodedInput]);
      const outputTensor = this.model.predict(inputTensor) as tf.Tensor2D;
      const prediction = outputTensor.argMax(-1);
      output = [...output, prediction.dataSync()[0]];
      encodedInput = [...encodedInput, prediction.dataSync()[0]];
    }

    return output;
  }

  private encodeWord(word: string) {
    const index = this.getVocabulary().indexOf(word);
    if (index === -1) {
      return this.getVocabulary().length;
    }
    return index;
  }

  private decodeWord(index: number) {
    if (index === this.getVocabulary().length) {
      return '<UNK>';
    }
    return this.getVocabulary()[index];
  }

  private getVocabulary() {
    if (!this.tokenizedData) {
      throw new Error(
        'Cannot get vocabulary: Model not trained or imported yet!'
      );
    }

    return [...new Set(this.tokenizedData.flat())];
  }

  private getVocabularySize() {
    return this.getVocabulary().length;
  }
}
