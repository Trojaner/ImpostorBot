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
  wordToIndex: {[word: string]: number};
  indexToWord: {[index: number]: string};
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
    const maxVocabSize = 40;

    const text = messages
      .map(msg => msg.replace(/\s{2,}/g, ' '))
      .filter(msg => msg.split(' ').length <= maxWordCount)
      .slice(0, maxMessages)
      .join(' \n');

    console.log('buildVocabulary');
    this.buildVocabulary(text, maxVocabSize);

    console.log('processData');
    const data = this.processData(text);
    const {inputSequences, targetSequences} = data;

    const model = tf.sequential();
    model.add(
      tf.layers.embedding({
        inputDim: this.tokenizedData!.vocabulary.size,
        outputDim: 16,
        batchSize,
        inputLength: inputSequences[0].length,
      })
    );
    model.add(
      tf.layers.lstm({
        units: 16,
        returnSequences: true,
        batchSize,
        recurrentInitializer: 'glorotNormal',
      })
    );
    model.add(
      tf.layers.dense({
        units: this.tokenizedData!.vocabulary.size,
        activation: 'softmax',
        batchSize,
      })
    );

    console.log('model.compile');
    model.compile({
      optimizer: 'adamax',
      loss: 'categoricalCrossentropy',
    });

    console.log('oneHotEncode');
    const oneHotInputs = this.oneHotEncode(inputSequences);
    const oneHotTargets = this.oneHotEncode(targetSequences);

    console.log('tensor3d');
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

    console.log('model.fit started');
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
    predictLengthMin: number = 10,
    predictLengthMax: number = 2000,
    temperature: number = 0.7
  ) {
    if (!this.model || !this.tokenizedData)
      throw new Error('Model not trained or imported yet.');

    if (inputText != '' && !inputText.endsWith(' ')) {
      inputText += ' ';
    }

    const words = inputText.split(' ');
    let oneHotInputs = this.oneHotEncode([inputText]);
    let inputTensor = tf.tensor3d(oneHotInputs, [
      1,
      oneHotInputs[0].length,
      this.tokenizedData!.vocabulary.size,
    ]);

    let prediction = '';
    let predictLength = predictLengthMin;

    while (prediction.split(' ').length < predictLengthMax) {
      const output = this.model.predict(inputTensor) as tf.Tensor<Rank.R3>;
      const logits = output.mul(tf.scalar(temperature)).softmax().flatten();
      const index = tf.multinomial(logits, 1).flatten().dataSync()[0];
      const word = this.tokenizedData!.indexToWord[index];

      prediction += `${word} `;
      words.shift();
      words.push(word);

      oneHotInputs = this.oneHotEncode([words.join(' ')]);
      inputTensor.dispose();
      inputTensor = tf.tensor3d(oneHotInputs, [
        1,
        oneHotInputs[0].length,
        this.tokenizedData!.vocabulary.size,
      ]);

      predictLength++;
    }

    inputTensor?.dispose();
    return words.join(' ');
  }

  generateRandom(
    minLength: number = 10,
    maxLength: number = 2000,
    temperature: number = 0.7
  ) {
    if (!this.model || !this.tokenizedData)
      throw new Error('Model not trained or imported yet.');

    return this.predictRemainder('', minLength, maxLength, temperature);
  }

  private buildVocabulary(text: string, maxVocabSize: number) {
    const vocabulary = new Set<string>();
    const wordToIndex: {[key: string]: number} = {};
    const indexToWord: {[key: number]: string} = {};

    const words = text.split(' ');
    const wordCount = new Map<string, number>();
    for (const word of words) {
      if (wordCount.has(word)) {
        wordCount.set(word, wordCount.get(word)! + 1);
      } else {
        wordCount.set(word, 1);
      }
    }

    const sortedWordCount = Array.from(wordCount.entries()).sort(
      (a, b) => b[1] - a[1]
    );

    for (let i = 0; i < maxVocabSize; i++) {
      const word = sortedWordCount[i][0];
      vocabulary.add(word);
      wordToIndex[word] = i;
      indexToWord[i] = word;
    }

    this.tokenizedData = {
      vocabulary,
      wordToIndex,
      indexToWord,
    };
  }

  processData(text: string) {
    const words = text.split(' ');
    const sequenceLength = 20;
    const inputSequences: string[] = [];
    const targetSequences: string[] = [];

    for (let i = 0; i < words.length - sequenceLength; i++) {
      const inputSequence = words.slice(i, i + sequenceLength).join(' ');
      const targetSequence = words[i + sequenceLength];

      inputSequences.push(inputSequence);
      targetSequences.push(targetSequence);
    }

    return {inputSequences, targetSequences};
  }

  private preprocessInput(input: string) {
    if (!this.tokenizedData) throw new Error('Tokenized data not found.');

    return input
      .split(' ')
      .map(char => this.tokenizedData!.wordToIndex[char] || '');
  }
  oneHotEncode(sequences: string[]): number[][][] {
    const oneHotEncodedSequences: number[][][] = [];

    for (const sequence of sequences) {
      const oneHotEncodedSequence: number[][] = [];

      for (const word of sequence.split(' ')) {
        const oneHotEncodedWord = Array<number>(
          this.tokenizedData!.vocabulary.size
        ).fill(0);
        oneHotEncodedWord[this.tokenizedData!.wordToIndex[word]] = 1;
        oneHotEncodedSequence.push(oneHotEncodedWord);
      }

      oneHotEncodedSequences.push(oneHotEncodedSequence);
    }

    return oneHotEncodedSequences;
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
