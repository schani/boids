const colorArray = [
  '#018AC0',
  '#0DD7D1',
  '#B265E9',
  '#F2BD7B',
  '#F7907D'
]

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const width = canvas.scrollWidth;
const height = canvas.scrollHeight;

const radius = 100;
const scale = 5;

const boids = [];
// for (let i = 0; i < 1000; i++) {
//   fs.push(new Follower(colorArray[i % 5], new Vector2(Math.random() * width, Math.random() * height), Math.random() * 2));
// }


for (let i = 0; i < 300; i++) {
  boids.push(new Boid(width * scale, height * scale, colorArray[i % 5]));
}

function updateASingleBoid(boid) {
  //
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

  boid.update(nearBoids);
  boid.draw(ctx);
}

function animate() {
  requestAnimationFrame(animate)

  ctx.save();
  ctx.clearRect(0, 0, width, height);
  ctx.scale(1 / scale, 1 / scale);

  let sum = new Vector2(0, 0);

  for (const boid of boids) {
    updateASingleBoid(boid);

    sum.add(boid.position);
  }

  sum.div(boids.length);
  ctx.fillStyle = "#ff0000";
  ctx.fillRect(sum.x - 30, sum.y - 30, 60, 60);

  ctx.restore();
}
animate()
