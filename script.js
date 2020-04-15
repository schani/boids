const colorArray = [
  "#A3A84F",
  "#FF718D",
  "#29CDFF",
  "#42E5E0",
  "#7A79FF",
  "#4467F8",
  "#E16AE3",
  // "#18A0AE",
  // "#E19A7A",
  // "#A659E3"
]

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { alpha: false });

const width = canvas.scrollWidth;
const height = canvas.scrollHeight;

const MAX_VELOCITY = 5;
const radius = 100;
const scale = 4;

let boids = [];

const drawCenterOfGravity = false;

const turbo = 1;

for (let i = 0; i < 3000; i++) {
  const mod = i % colorArray.length;
  const color = colorArray[mod];
  const targetVelocity = MAX_VELOCITY; //* (mod + 1) /  colorArray.length;

  boids.push(new Boid(width * scale, height * scale, color, targetVelocity));
}

class DefaultArray2D {
  constructor(makeDefault) {
    this.makeDefault = makeDefault;
    this.arr = []
  }

  get(x, y) {
    let row = this.arr[x];
    if (row === undefined) {
      row = this.arr[x] = [];
    }
    let cell = row[y];
    if (cell === undefined) {
      cell = row[y] = this.makeDefault();
    }
    return cell;
  }


}

function boidSlice(b) {
  let x = b.position.x;
  let y = b.position.y;
  let sliceX = Math.max(0, Math.floor(x / radius));
  let sliceY = Math.max(0, Math.floor(y / radius));

  return { x: sliceX, y: sliceY };
}

function slice(boids) {
  let sliceArray = new DefaultArray2D(() => []);
  for (const b of boids) {
    let { x, y } = boidSlice(b);
    sliceArray.get(x, y).push(b);
  }

  return sliceArray;
}

function calculateSingleBoid(boid, sliceArray) {
  // get the list of all the nearby boids and store in nearBoids
  const nearBoids = [];

  let { x, y } = boidSlice(boid);
  let boidsInSlice = [];

  for (const dx of [-1, 0, 1]) {
    for (const dy of [-1, 0, 1]) {
      const sx = x + dx;
      const sy = y + dy;

      for (const b of sliceArray.get(sx, sy)) {
        if (b !== boid) {
          let dist = Vector2.dist(b.position, boid.position);
          if (dist < radius) {
            nearBoids.push(b);
          }
        }
      }
    }
  }

  return boid.calculate(nearBoids);
}

let lastFrameTime = Date.now();

function animate() {
  requestAnimationFrame(animate)

  const now = Date.now();
  const frameDuration = now - lastFrameTime;
  lastFrameTime = now;

  for (let i = 0; i < turbo; i++) {
    const newBoids = [];
    const sliceArray = slice(boids);
    for (const boid of boids) {
      newBoids.push(...calculateSingleBoid(boid, sliceArray));
    }
    boids = newBoids;
    for (const boid of boids) {
      boid.update();
    }
  }

  ctx.save();

  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "black";
  ctx.font = "16px serif";
  ctx.fillText(boids.length.toString(), width - 50, height - 5);

  ctx.fillText(Math.floor(1000 / frameDuration).toString(), 5, height - 5);



  ctx.scale(1 / scale, 1 / scale);

  let sum = new Vector2(0, 0);
  for (const color of colorArray) {
    ctx.beginPath()
    for (const boid of boids) {
      if (boid.color !== color) continue;
      boid.draw(ctx);
      sum.add(boid.position);
    }
    ctx.fillStyle = color;
    ctx.fill()
  }

  if (drawCenterOfGravity) {
    sum.div(boids.length);
    ctx.fillStyle = "#ff0000";
    ctx.fillRect(sum.x - 30, sum.y - 30, 60, 60);
  }

  ctx.restore();
}
animate()
