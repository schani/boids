const colorArray = [
  '#018AC0',
  '#0DD7D1',
  '#B265E9',
  '#F2BD7B',
  '#F7907D'
]

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let mouse = undefined;

canvas.addEventListener('mousemove',
  (event) => {
    mouse = new Vector2(event.x, event.y);
  })

const width = canvas.scrollWidth;
const height = canvas.scrollHeight;

const fs = [];
for (let i = 0; i < 1000; i++) {
  fs.push(new Follower(colorArray[i % 5], new Vector2(Math.random() * width, Math.random() * height), Math.random() * 2));
}

function animate() {
  requestAnimationFrame(animate)

  ctx.clearRect(0, 0, width, height);

  for (const f of fs) {
    f.update(mouse);
    f.draw(ctx);
  }
}
animate()
