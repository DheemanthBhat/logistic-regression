var trainingFeatures = [];
var trainingLabels = [];
var testingFeatures = [];
var testingLabels = [];
var input = [];

for (let i = 0; i < featureSet1.length; i++) {
  trainingFeatures.push([
    featureSet1[i][0],
    featureSet1[i][1],
    // Math.pow(featureSet1[i][0], 1 / 2),
    // Math.pow(featureSet1[i][1], 1 / 2)
  ])
}

for (let i = 0; i < featureSet2.length; i++) {
  testingFeatures.push([
    featureSet2[i][0],
    featureSet2[i][1],
    // Math.pow(featureSet2[i][0], 1 / 2),
    // Math.pow(featureSet2[i][1], 1 / 2)
  ])
}

for (let i = 0; i < labelSet1.length; i++) {
  let label = labelSet1[i][0] == 'A' ? 1 : 0;
  trainingLabels.push([label]);
}

for (let i = 0; i < labelSet2.length; i++) {
  let label = labelSet2[i][0] == 'A' ? 1 : 0;
  testingLabels.push([label]);
}


let battery = tf.linspace(1, 100, 100).arraySync();
let gsm = tf.linspace(1, 100, 100).arraySync()

for (let i = 1; i <= 100; i++) {
  for (let j = 1; j <= 100; j++) {
    input.push([i, j]);
  }
}