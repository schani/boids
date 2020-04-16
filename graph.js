const graph = document.getElementById("graph");
const graphCtx = graph.getContext("2d", { alpha: false });

const graphWidth = graph.scrollWidth;
const graphHeight = graph.scrollHeight;

const maxGraphScale = 320;

class Graph {
  constructor(color) {
    this.color = color;
    this.history = [];
    this.max = 0;
    this.scale = 1;
    this.resetAcc();
  }

  resetAcc() {
    this.acc = 0;
    this.accCount = 0;
  }

  addToHistory(y) {
    this.history.push(y);

    if (this.history.length > graphWidth) {
      if (this.scale < maxGraphScale) {
        const newHistory = [];
        for (let i = 0; i < this.history.length - 1; i += 2) {
          newHistory.push((this.history[i] + this.history[i+1]) / 2);
        }
        this.history = newHistory;
        this.scale *= 2;
      } else {
        while (this.history.length > graphWidth) {
          this.history.shift();
        }
      }
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

    if (this.accCount < this.scale) return false;

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