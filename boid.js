let mouse = undefined;

const startLife = 300;

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

  align(boids) {
    let steer = new Vector2(0, 0);
    let n = 0

    for (const boid of boids) {
      if (this.color === boid.color) {
        steer.add(boid.velocity);
        n++
      }
    }

    if (n === 0) {
      return steer;
    }

    steer.div(n);

    let align = steer.sub(this.velocity);
    return Vector2.norm(align).div(4);
  }

  cohesion(boids) {
    let steer = new Vector2(0, 0);
    let n = 0

    for (const boid of boids) {
      if (this.color === boid.color) {
        steer.add(boid.position);
        n++
      }
    }

    if (n === 0) {
      return steer;
    }

    steer.div(n);

    let cohesion = steer.sub(this.position);
    return Vector2.norm(cohesion).div(4);

  }

  separation(boids, separationFactor) {
    let finalVec = new Vector2();

    for (const boid of boids) {
      let diff = Vector2.sub(this.position, boid.position);
      let len = diff.length();
      diff.norm();
      diff.mul(1 / len * separationFactor);

      finalVec.add(diff);
    }

    return finalVec;
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

  updateLife(boids) {
    let n = 0;
    for (const boid of boids) {
      if (this.color === boid.color) {
        n += 1;
      }
    }
    if (n < 5 || n > 17) {
      this.life -= 1;
    } else {
      this.life += 1;
    }
  }

  calculate(boids) {
    const diff = (this.targetVelocity - this.velocity.length());
    this.nextVelocity = Vector2.add(this.velocity, Vector2.norm(this.velocity).mul(diff * 0.2));

    this.nextVelocity.add(this.align(boids));

    this.nextVelocity.add(this.cohesion(boids));
    this.nextVelocity.add(this.separation(boids, 3));

    // if (mouse !== undefined) {
    //   this.velocity.add(this.separation([mouse], 70));
    // }

    this.stayInBounds();

    this.updateLife(boids);

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
