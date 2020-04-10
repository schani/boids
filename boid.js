const MAX_VELOCITY = 7;

let mouse = undefined;

const c = document.getElementById('canvas');
c.addEventListener('mousemove',
  (event) => {
    mouse = { position: new Vector2(event.x * 5, event.y * 5) };
  })


class Boid {
  constructor(width, height, color) {
    this.width = width;
    this.height = height;

    this.position = new Vector2(Math.random() * width, Math.random() * height);
    this.velocity = new Vector2(Math.random() - 0.5, Math.random() - 0.5).mul(10);

    this.color = color;

    this.targetVelocity = MAX_VELOCITY * Math.random();
  }


  align(boids) {
    let steer = new Vector2(0, 0);
    let doSteer = false;

    for (const boid of boids) {
      steer.add(boid.velocity);
      doSteer = true;
    }

    if (!doSteer) {
      return new Vector2(0, 0);
    }

    steer.div(boids.length);

    let align = steer.sub(this.velocity);
    return Vector2.norm(align).div(4);
  }

  cohesion(boids) {
    let steer = new Vector2(0, 0);
    let doSteer = false;

    for (const boid of boids) {
      // if (this.color === boid.color) {
        steer.add(boid.position);
        doSteer = true;
      // }
    }

    if (!doSteer) {
      return new Vector2(0, 0);
    }

    steer.div(boids.length);

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
      this.velocity.x = Math.abs(this.velocity.x);
    } else if (this.position.x > this.width) {
      this.velocity.x = -Math.abs(this.velocity.x);
    }

    if (this.position.y < 0) {
      this.velocity.y = Math.abs(this.velocity.y);
    } else if (this.position.y > this.height) {
      this.velocity.y = -Math.abs(this.velocity.y);
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
