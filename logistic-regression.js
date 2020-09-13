/**
 * Vectorized implementation of Linear-regression with Gradient-descent
 */
class LogisticRegression {
  options = {
    learningRate: 0.1,
    iterations: 1000,
    batchSize: 10,
    decisionBoundary: 0.5
  };

  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);

    this.n = this.features.shape[0];
    this.c = this.features.shape[1];
    this.weights = tf.zeros([this.c, 1]);

    this.options = { ...this.options, ...options };
    this.crossEntropies = [];
  }

  processFeatures(featuresArray) {
    let features = tf.tensor(featuresArray);
    features = this.standardize(features);

    let n = features.shape[0];
    let identity = tf.ones([n, 1]); // Identity vector
    // Concat features to identity vector.
    features = identity.concat(features, 1);

    return features;
  }

  normalEquation() {
    let xTranspose = this.features.transpose();

    let A = xTranspose.matMul(this.features);

    let AInverse = tf.tensor(math.inv(A.arraySync()));

    let theta = AInverse.matMul(xTranspose).matMul(this.labels);

    return theta;
  }

  // Mean Normalization
  standardize(features) {
    if (this.mean === undefined || this.standardDeviation === undefined) {
      let { mean, variance } = tf.moments(features, 0);
      this.mean = mean;
      this.standardDeviation = variance.pow(0.5);
    }

    return features.sub(this.mean).div(this.standardDeviation);
  }

  train() {
    console.log(`${this.n} records are used for training.`);
    let batchCount = parseInt(this.n / this.options.batchSize);
    console.log(`Batch Count: ${batchCount}`);
    for (let i = 0; i < this.options.iterations; i++) {
      // console.log(`Learning rate: ${this.options.learningRate}`);
      for (let j = 0; j < batchCount; j++) {
        let { features, labels } = this.getNextBatch(j);
        this.weights = this.gradientDescent(features, labels);
      }

      this.recordCost();
      this.updateLearningRate();
    }
  }

  gradientDescent(features, labels) {
    // Total number of rows in features.
    let n = features.shape[0];

    /**
     * X*Theta = t0 + t1*x1 + t2*x2 + ... + tn*xn
     * h = 1 / (1 + e ^ -(X * Theta))                // Vector implementation
     * Where:
     *  h - Hypothesis,
     *  x - features,
     *  X - feature matrix,
     *  Theta - parameters/weights matrix.
     */
    let h = features.matMul(this.weights).sigmoid();

    let difference = h.sub(labels);

    // Derivative of Cost or J(Theta) w.r.t weights:
    let slopes = features
      .transpose()
      .matMul(difference)
      //.mul(2) // Optional (Usually this is omitted in most of the Gradient descent implementations).
      .div(n);

    /**
     * Multiply slopes with learning rate
     * and subtract results from weights.
     */
    return this.weights.sub(slopes.mul(this.options.learningRate));
  }

  test(testFeaturesArray, testLabelsArray) {
    let testLabels = tf.tensor(testLabelsArray);
    let h = this.predict(testFeaturesArray);
    let totalRows = h.shape[0];
    let incorrectPredictions = h.sub(testLabels).abs().sum().arraySync();
    // Calculate accuracy of the model.
    return (totalRows - incorrectPredictions) / totalRows * 100;
  }

  // Cost function
  recordCost() {
    let h = this.features.matMul(this.weights).sigmoid();
    let LHS = tf.scalar(-1).mul(this.labels).transpose().matMul(h.log());
    let RHS = tf.scalar(1).sub(this.labels).transpose().matMul(tf.scalar(1).sub(h).log());
    let cost = LHS.sub(RHS).div(this.n).dataSync()[0];
    this.crossEntropies.unshift(cost);
  }

  updateLearningRate() {
    if (this.crossEntropies.length < 2) return;

    let currCost = this.crossEntropies[0];
    let prevCost = this.crossEntropies[1];
    if (currCost > prevCost) {
      // if Cost went up, divide learning rate by 2.
      this.options.learningRate /= 2;
    } else {
      // if Cost went down, increase learning rate by 5%.
      this.options.learningRate *= 1.05;
    }
  }

  getNextBatch(j) {
    let { batchSize } = this.options;
    let startIndex = j * batchSize;

    let features = this.features.slice([startIndex, 0], [batchSize, -1]);
    let labels = this.labels.slice([startIndex, 0], [batchSize, -1]);

    return { features, labels };
  }

  predict(observations) {
    let processedFeatures = this.processFeatures(observations);
    return processedFeatures
      .matMul(this.weights)
      .sigmoid()
      .greaterEqual(this.options.decisionBoundary)
      .cast("float32");
  }
}
