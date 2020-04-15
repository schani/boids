let mouse = undefined;

const startLife = 300;
const separationCoefficient = 3;

const c = document.getElementById('canvas');
c.addEventListener('mousemove',
  (event) => {
    mouse = { position: new Vector2(event.x * 5, event.y * 5) };
  })


class Boid {
  constructor(width, height, color, targetVelocity) {
    this.width = width;
    this.height = height;

    this.position = new Vector2(Math.random() * width, Math.random() * height);
    this.velocity = new Vector2(Math.random() - 0.5, Math.random() - 0.5).mul(10);

    this.color = color;

    this.targetVelocity = targetVelocity;

    this.life = startLife;
  }

  split() {
    const b = new Boid(this.width, this.height, this.color, this.targetVelocity);
    b.position = new Vector2(this.position.x + 3, this.position.y + 3);
    b.velocity = new Vector2(this.velocity.x, this.velocity.y);
    b.life = this.life = this.life / 2;
    if (this.nextPosition !== undefined) {
      b.nextPosition = new Vector2(this.nextPosition.x + 3, this.nextPosition.y + 3);
    }
    if (this.nextVelocity != undefined) {
      b.nextVelocity = new Vector2(this.nextVelocity.x, this.nextVelocity.y);

    }
    return [this, b];
  }

  stayInBounds() {
    if (this.position.x < 0) {
      this.nextVelocity.x = Math.abs(this.nextVelocity.x);
    } else if (this.position.x > this.width) {
      this.nextVelocity.x = -Math.abs(this.nextVelocity.x);
    }

    if (this.position.y < 0) {
      this.nextVelocity.y = Math.abs(this.nextVelocity.y);
    } else if (this.position.y > this.height) {
      this.nextVelocity.y = -Math.abs(this.nextVelocity.y);
    }
  }

  calculate(boids) {
    const diff = (this.targetVelocity - this.velocity.length());
    this.nextVelocity = Vector2.add(this.velocity, Vector2.norm(this.velocity).mul(diff * 0.2));

    let alignSteer = new Vector2(0, 0);
    let cohesionSteer = new Vector2(0, 0);
    let separationSteer = new Vector2(0, 0);
    let numFriends = 0;

    for (const boid of boids) {
      // separation
      const diff = Vector2.sub(this.position, boid.position);
      const len = diff.length();
      diff.norm();
      diff.mul(1 / len * separationCoefficient);
      separationSteer.add(diff);

      if (this.color === boid.color) {
        // alignment
        alignSteer.add(boid.velocity);

        // cohesion
        cohesionSteer.add(boid.position);

        numFriends++
      }
    }

    // separation
    this.nextVelocity.add(separationSteer);

    // alignment
    if (numFriends > 0) {
      alignSteer.div(numFriends);
      const align = alignSteer.sub(this.velocity);
      this.nextVelocity.add(Vector2.norm(align).div(4));
    }

    // cohesion
    if (numFriends > 0) {
      cohesionSteer.div(numFriends);
      const cohesion = cohesionSteer.sub(this.position);
      this.nextVelocity.add(Vector2.norm(cohesion).div(4));
    }

    this.stayInBounds();

    // update life
    if (numFriends < 5 || numFriends > 17) {
      this.life -= 1;
    } else {
      this.life += 1;
    }

    if (this.life > startLife * 2) {
      return this.split();
    } else if (this.life > 0) {
      return [this];
    } else {
      return [];
    }
  }

  update() {
    this.velocity = this.nextVelocity;
    this.position.add(this.velocity);
  }

  draw(ctx) {
    const angleRad = Math.atan2(this.velocity.y, this.velocity.x);

    const sin = Math.sin(angleRad);
    const cos = Math.cos(angleRad);

    function rotatePoint(x, y, position) {
      return [(x * cos - y * sin) + position.x, (x * sin + y * cos) + position.y]
    }

    const start = rotatePoint(20, 0, this.position)
    ctx.moveTo(...start);
    ctx.lineTo(...rotatePoint(-7, 7, this.position));
    ctx.lineTo(...rotatePoint(-7, -7, this.position));
    ctx.lineTo(...start);
  }
}
