const colorArray = [
  '#018AC0',
  '#0DD7D1',
  '#B265E9',
  // '#F2BD7B',
  // '#F7907D'
]

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const width = canvas.scrollWidth;
const height = canvas.scrollHeight;

const MAX_VELOCITY = 7;
const radius = 100;
const scale = 4;

const boids = [];

const drawCenterOfGravity = false;

const turbo = 1;

for (let i = 0; i < 1000; i++) {
  const mod = i % colorArray.length;
  const color = colorArray[mod];
  const targetVelocity = MAX_VELOCITY; //* (mod + 1) /  colorArray.length;

  boids.push(new Boid(width * scale, height * scale, color, targetVelocity));
}

function calculateSingleBoid(boid) {
  // get the list of all the nearby boids and store in nearBoids
  const nearBoids = [];

  for (const b of boids) {
    if (b !== boid) {
      let dist = Vector2.dist(b.position, boid.position);
      if (dist < radius) {
        nearBoids.push(b);
      }
    }
  }

  boid.calculate(nearBoids);
}

let lastFrameTime = Date.now();

function animate() {
  requestAnimationFrame(animate)

  const now = Date.now();
  const frameDuration = now - lastFrameTime;
  lastFrameTime = now;

  for (let i = 0; i < turbo; i++) {
    for (const boid of boids) {
      calculateSingleBoid(boid);
    }
    for (const boid of boids) {
      boid.update();
    }
  }

  ctx.save();

  ctx.clearRect(0, 0, width, height);

  ctx.font = "16px serif";
  ctx.fillText(Math.floor(1000 / frameDuration).toString(), 5, height - 5);

  ctx.scale(1 / scale, 1 / scale);

  let sum = new Vector2(0, 0);
  for (const boid of boids) {
    boid.draw(ctx);
    sum.add(boid.position);
  }

  if (drawCenterOfGravity) {
    sum.div(boids.length);
    ctx.fillStyle = "#ff0000";
    ctx.fillRect(sum.x - 30, sum.y - 30, 60, 60);
  }

  ctx.restore();
}
animate()
