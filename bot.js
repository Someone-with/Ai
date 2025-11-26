// Tiny character-level RNN
const vocab = 'abcdefghijklmnopqrstuvwxyz0123456789 .!?';
const vocabSize = vocab.length;
const seqLen = 40;
const hidden = 64;

let model;

function buildModel() {
  model = tf.sequential();
  model.add(tf.layers.lstm({ units: hidden, inputShape: [seqLen, vocabSize], returnSequences: true }));
  model.add(tf.layers.timeDistributed({ layer: tf.layers.dense({ units: vocabSize, activation: 'softmax' }) }));
  model.compile({ loss: 'categoricalCrossentropy', optimizer: tf.train.rmsprop(0.01) });
}

function textToTensor(text) {
  const idx = c => vocab.indexOf(c.toLowerCase()) >= 0 ? vocab.indexOf(c.toLowerCase()) : 0;
  const oneHot = i => Array(vocabSize).fill(0).map((_, j) => j === i ? 1 : 0);
  const xs = [], ys = [];
  for (let i = 0; i <= text.length - seqLen - 1; i++) {
    xs.push([...text.slice(i, i + seqLen)].map(c => oneHot(idx(c))));
    ys.push([...text.slice(i + 1, i + seqLen + 1)].map(c => oneHot(idx(c))));
  }
  return { xs: tf.tensor3d(xs), ys: tf.tensor3d(ys) };
}

const data = [
  "Q: What is gravity? A: Gravity is a force that attracts objects toward each other.",
  "Q: How do I design a chair? A: Start with a stable base, then add a seat and backrest.",
  "Q: What is 2+2? A: 4",
  "Q: Who wrote Romeo and Juliet? A: William Shakespeare.",
  "Q: What is the capital of France? A: Paris."
];

async function train() {
  buildModel();
  const { xs, ys } = textToTensor(data.join(' '));
  await model.fit(xs, ys, { epochs: 50, verbose: 1 });
  await model.save('downloads://model');
  alert("Training done! Download model.json and commit it to repo.");
}

function sample(preds) {
  const p = Array.from(preds);
  const i = p.indexOf(Math.max(...p));
  return vocab[i];
}

async function send() {
  if (!model) {
    try {
      model = await tf.loadLayersModel('model.json');
    } catch {
      document.getElementById('out').textContent = "Model not found. Press 'Train Model' first.";
      return;
    }
  }
  const prompt = document.getElementById('in').value.toLowerCase();
  let seed = prompt.slice(-seqLen).padStart(seqLen, ' ');
  let out = seed;
  for (let k = 0; k < 80; k++) {
    const input = [...seed].map(c => {
      const idx = vocab.indexOf(c);
      return idx >= 0 ? idx : 0;
    });
    const tensor = tf.tensor3d([input.map(i => Array(vocabSize).fill(0).map((_, j) => j === i ? 1 : 0))]);
    const preds = model.predict(tensor).squeeze().argMax(-1).dataSync();
    const next = vocab[preds[preds.length - 1]];
    out += next;
    seed = seed.slice(1) + next;
  }
  document.getElementById('out').textContent = out;
}
