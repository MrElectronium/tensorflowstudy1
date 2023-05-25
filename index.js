const tf = require('@tensorflow/tfjs-node-gpu');

function Normalize(num,max)
{
return Number(num)/max;
}
function NormalizeArr(arr,max)
{
return arr.map(x=>x/max);
}
// Eğitim verilerini oluştur
function generateTrainingData(arr) {
  const input = [];
  const output = [];
  const len=arr.length;
  const max=Max(arr);

  for (let i = 0; i < len-5; i++) {
    //Aşağıdaki tüm değerleri max'a bölerek normalize ediyoruz
    const x1 = arr[i]/max;
    const x2 =arr[i+1]/max;
    const x3 = arr[i+3]/max;
    const x4 = arr[i+4]/max;
    const y = arr[i+2]/max; // 

    input.push([x1,x2,x3,x4]);
    output.push(y);
  }

  return { input: tf.tensor2d(input), output: tf.tensor1d(output) };
}

// Modeli oluştur
function createModel() {
  const model = tf.sequential();
  
  // Giriş katmanı
  model.add(tf.layers.dense({ units: 8, inputShape: [4], activation: 'relu' }));

  // Gizli katmanlar
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));

  // Çıkış katmanı
  model.add(tf.layers.dense({ units: 1 }));

  return model;
}

// Modeli eğit
async function trainModel(model, input, output) {
  model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

  const history = await model.fit(input, output, {
    epochs: 50,
    verbose: 0
  });

  return history;
}
function Max(arr)
{
    var max=0;
    for(let i=0;i<arr.length;i++)
    {
    if(arr[i]>max)
    max=arr[i];
    }
    return max;
}
// Modeli kullanarak tahmin yap
function predict(model, input) {
  const prediction = model.predict(input);
  return prediction.dataSync()[0];
}
function normalizeValue(value, min, max) {
    return (value - min) / (max - min);
  }
// Örnek kullanım
async function run() {
   
    var data=[];
    for(let i=0;i<100;i++)
    {
    data.push(Number(i)*5);
    }
  const trainingData = generateTrainingData(data);
  const model = createModel();

  console.log('Model eğitiliyor...');
  const history = await trainModel(model, trainingData.input, trainingData.output);
  console.log('Eğitim tamamlandı.');
  const max=Max(data);
   let input=[5,10,20,25].map(x=>x/max);
  
  const testInput = tf.tensor2d([input]);
  const prediction = predict(model, testInput);

  console.log('Tahmin:', prediction*max);

}

run();
