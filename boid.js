let mouse = undefined;

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
      return new Vector2(0, 0);
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
      return new Vector2(0, 0);
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

  calculate(boids) {
    const diff = (this.targetVelocity - this.velocity.length());
    this.nextVelocity = Vector2.add(this.velocity,Vector2.norm(this.velocity).mul(diff * 0.2) );

    this.nextVelocity.add(this.align(boids));

    this.nextVelocity.add(this.cohesion(boids));
    this.nextVelocity.add(this.separation(boids, 3));

    // if (mouse !== undefined) {
    //   this.velocity.add(this.separation([mouse], 70));
    // }

    this.stayInBounds();
  }

  update() {
    this.velocity = this.nextVelocity;
    this.position.add(this.velocity);
  }

  draw(ctx) {
    ctx.fillStyle = this.color;
    ctx.save();

    ctx.translate(this.position.x, this.position.y);
    const angleRad = Math.atan2(this.velocity.y, this.velocity.x);
    ctx.rotate(angleRad)

    ctx.beginPath();
    ctx.moveTo(20, 0);
    ctx.lineTo(-7, -7);
    ctx.lineTo(-7, 7);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }
}
