const graph = document.getElementById("graph");
const graphCtx = graph.getContext("2d", { alpha: false });

const graphWidth = graph.scrollWidth;
const graphHeight = graph.scrollHeight;

const graphScale = 20;

class Graph {
  constructor(color) {
    this.color = color;
    this.history = [];
    this.max = 0;
    this.resetAcc();
  }

  resetAcc() {
    this.acc = 0;
    this.accCount = 0;
  }

  addToHistory(y) {

    this.history.push(y);
    while (this.history.length > graphWidth) {
      this.history.shift();
    }
    this.max = Math.max(this.max, y);

    // console.log(this.color, y, JSON.stringify(this.history));;
  }

  draw() {
    graphCtx.strokeStyle = this.color;
    graphCtx.beginPath();
    for (let i = 0; i < this.history.length; i++) {
      const x = graphWidth - this.history.length + i;
      const y = graphHeight * (1 - this.history[i] / this.max);
      if (i === 0) {
        graphCtx.moveTo(x, y);
      } else {
        graphCtx.lineTo(x, y);
      }
    }
    graphCtx.stroke();
  }

  addPoint(y) {
    this.acc += y;
    this.accCount++;

    if (this.accCount < graphScale) return false;

    this.addToHistory(this.acc / this.accCount);
    this.resetAcc();
    return true;
  }
}

let fishGraph = new Graph("blue");
let sharkGraph = new Graph("red");

function drawGraph(numFish, numShark) {
  const drawFish = fishGraph.addPoint(numFish);
  const drawShark = sharkGraph.addPoint(numShark);

  if (drawFish || drawShark) {
    graphCtx.save();

    graphCtx.fillStyle = "white";
    graphCtx.fillRect(0, 0, graphWidth, graphHeight);

    fishGraph.draw();
    sharkGraph.draw();

    graphCtx.restore();
  }
}