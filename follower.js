const wobble = 5;

class Follower {
  constructor(c, p, l) {
    this.c = c;
    this.p = p;
    this.l = l;
    this.v = new Vector2(0, 0);
  }

  update(m) {
    this.v = Vector2.mul(this.v, 0.8);
    if (m !== undefined) {
      let dv = Vector2.sub(m, this.p).norm().mul(this.l);
      this.v = Vector2.add(this.v, dv);
    }
    this.v = Vector2.add(this.v, new Vector2(Math.random() - 0.5, Math.random() - 0.5).mul(wobble));
    this.p = Vector2.add(this.p, this.v);
  }

  draw(ctx) {
    ctx.fillStyle = this.c;
    ctx.save();

    ctx.translate(this.p.x, this.p.y);
    const angleRad = Math.atan2(this.v.y, this.v.x);
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
