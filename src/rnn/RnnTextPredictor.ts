import * as tf from '@tensorflow/tfjs-node';
import {io as tfio} from '@tensorflow/tfjs';
import {Tensor, Rank} from '@tensorflow/tfjs-node';

export type ExportedModel = {
  modelTopology: any;
  weightSpecs: tfio.WeightsManifestEntry[];
  weightData: ArrayBuffer;
  tokenizedData: TokenizedData;
};

export type TokenizedData = {
  vocabulary: Set<string>;
  charToIndex: {[word: string]: number};
  indexToChar: {[index: number]: string};
};

export default class RnnTextPredictor {
  model: tf.LayersModel | null = null;
  tokenizedData: TokenizedData | null = null;

  constructor() {}

  async train(messages: string[]) {
    const maxMessages = 512;
    const maxWordCount = 20;
    const batchSize = 8;
    const epochs = 5;

    const text = messages
      .map(msg => msg.replace(/\s{2,}/g, ' '))
      .filter(msg => msg.split(' ').length <= maxWordCount)
      .slice(0, maxMessages)
      .join(' \n');

    this.buildVocabulary(text);

    const data = this.processData(text);
    const {inputSequences, targetSequences} = data;

    const model = tf.sequential();
    model.add(
      tf.layers.embedding({
        inputDim: this.tokenizedData!.vocabulary.size,
        outputDim: 32,
        inputLength: inputSequences[0].length,
      })
    );
    model.add(
      tf.layers.lstm({
        units: 32,
        returnSequences: true,
        recurrentInitializer: 'glorotNormal',
      })
    );
    model.add(
      tf.layers.dense({
        units: this.tokenizedData!.vocabulary.size,
        activation: 'softmax',
      })
    );

    model.compile({
      optimizer: 'adamax',
      loss: 'categoricalCrossentropy',
    });

    const oneHotInputs = this.oneHotEncode(inputSequences);
    const oneHotTargets = this.oneHotEncode(targetSequences);

    const inputTensor = tf.tensor3d(oneHotInputs, [
      oneHotInputs.length,
      oneHotInputs[0].length,
      this.tokenizedData!.vocabulary.size,
    ]);
    const targetTensor = tf.tensor3d(oneHotTargets, [
      oneHotTargets.length,
      oneHotTargets[0].length,
      this.tokenizedData!.vocabulary.size,
    ]);

    await model.fit(inputTensor, targetTensor, {
      epochs,
      callbacks: {
        onBatchEnd: async (batch, log) => {
          console.log(`Batch ${batch}: loss = ${log?.loss}`);
        },
        onEpochEnd: async (epoch, log) => {
          console.log(`Epoch ${epoch}: loss = ${log?.loss}`);
        },
      },
    });

    this.model = model;

    inputTensor.dispose();
    targetTensor.dispose();
  }

  predictRemainder(
    inputText: string,
    numChars: number,
    temperature: number = 0.7
  ) {
    if (!this.model || !this.tokenizedData)
      throw new Error('Model not trained or imported yet.');

    inputText += ' ';

    const inputSequence = this.preprocessInput(inputText);
    const oneHotInput = this.oneHotEncode([inputSequence])[0];
    let inputTensor = tf.tensor3d(oneHotInput, [
      1,
      inputSequence.length,
      this.tokenizedData.vocabulary.size,
    ]);

    let predictedText = inputText;
    for (let i = 0; i < numChars - inputText.length; i++) {
      const prediction = this.model.predict(inputTensor) as Tensor;
      const index = this.sampleFromPrediction(
        tf.squeeze(prediction),
        temperature
      );
      const char = this.tokenizedData.indexToChar[index];
      predictedText += char;
      inputSequence.push(index);
      inputTensor.dispose();

      inputTensor = tf.tensor3d(this.oneHotEncode([inputSequence])[0], [
        1,
        inputSequence.length,
        this.tokenizedData.vocabulary.size,
      ]);
    }

    inputTensor?.dispose();
    return predictedText;
  }

  generateRandom(length: number = 2000) {
    if (!this.model || !this.tokenizedData)
      throw new Error('Model not trained or imported yet.');

    return this.predictRemainder('', length);
  }

  private buildVocabulary(text: string) {
    this.tokenizedData = {
      vocabulary: new Set<string>(),
      charToIndex: {},
      indexToChar: {},
    };

    for (let i = 0; i < text.length; i++) {
      const char = text[i];
      this.tokenizedData.vocabulary.add(char);
      this.tokenizedData.charToIndex[char] = i;
      this.tokenizedData.indexToChar[i] = char;
    }
  }

  private processData(text: string) {
    if (!this.tokenizedData) throw new Error('Tokenized data not found.');

    const inputSequences: number[][] = [];
    const targetSequences: number[][] = [];

    const sequenceLength = 100;
    for (let i = 0; i < text.length - sequenceLength; i++) {
      const inputSequence = text.substring(i, i + sequenceLength);
      const targetChar = text[i + sequenceLength];

      inputSequences.push(this.preprocessInput(inputSequence));
      targetSequences.push([this.tokenizedData.charToIndex[targetChar]]);
    }

    return {
      inputSequences,
      targetSequences,
    };
  }

  private preprocessInput(input: string) {
    if (!this.tokenizedData) throw new Error('Tokenized data not found.');

    return input.split('').map(char => this.tokenizedData!.charToIndex[char]);
  }

  private oneHotEncode(sequences: number[][]) {
    if (!this.tokenizedData) throw new Error('Tokenized data not found.');

    return sequences.map(sequence => {
      return sequence.map(index => {
        const oneHot = Array(this.tokenizedData!.vocabulary.size).fill(0);
        oneHot[index] = 1;
        return oneHot;
      });
    });
  }

  private sampleFromPrediction(prediction: Tensor, temperature: number) {
    if (!this.tokenizedData) throw new Error('Tokenized data not found.');

    return tf.tidy(() => {
      const logits = tf.div(tf.log(prediction), Math.max(temperature, 1e-6));
      const isNormalized = false;
      // `logits` is for a multinomial distribution, scaled by the temperature.
      // We randomly draw a sample from the distribution.
      return tf
        .multinomial(logits as any, 1, undefined, isNormalized)
        .dataSync()[0];
    });
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
}
