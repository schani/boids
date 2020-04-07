const MAX_VELOCITY = 7;

class Boid {
  constructor(width, height, color) {
    this.width = width;
    this.height = height;

    this.position = new Vector2(Math.random() * width, Math.random() * height);
    this.velocity = new Vector2(Math.random() - 0.5, Math.random() - 0.5);
    this.velocity.mul(10);
    // this.acceleration = acceleration;
    this.color = color;
  }


  align(boids) {
    let steer = new Vector2(0,0);

    for (const boid of boids) {
      steer.add(boid.velocity);
    }

    steer.div(boids.length);
    
    let align = steer.sub(this.velocity);
    return Vector2.norm(align).div(4);
  }

  cohesion(boids) {
  let steer = new Vector2(0,0);

    for (const boid of boids) {
      steer.add(boid.position);
    }

    steer.div(boids.length);
    
    let cohesion = steer.sub(this.position);
    return Vector2.norm(cohesion).div(4);

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

  update(boids) {
    const diff = (MAX_VELOCITY - this.velocity.length());
    this.velocity.add(Vector2.norm(this.velocity).mul(diff * 0.2));

    this.velocity.add(this.align(boids));
    this.velocity.add(this.cohesion(boids));
    this.position.add(this.velocity);

    this.stayInBounds();
  }

  draw(ctx) {
    ctx.fillStyle = this.color;
    ctx.save();

    ctx.translate(this.position.x, this.position.y);
    const angleRad = Math.atan2(this.velocity.y, this.velocity.x);
    ctx.rotate(angleRad)

    ctx.beginPath();
    ctx.moveTo(15, 0);
    ctx.lineTo(-5, -5);
    ctx.lineTo(-5, 5);
    ctx.closePath();
    ctx.fill();

    ctx.restore();
  }
}
