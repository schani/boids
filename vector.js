class Vector2 {
  constructor(x, y) {
    this.x = x || 0;
    this.y = y || 0;
  }

  neg() {
    this.x = -this.x;
    this.y = -this.y;
    return this;
  }

  add(v) {
    if (v instanceof Vector2) {
      this.x += v.x;
      this.y += v.y;
    } else {
      this.x += v;
      this.y += v;
    }
    return this;
  }

  sub(v) {
    if (v instanceof Vector2) {
      this.x -= v.x;
      this.y -= v.y;
    } else {
      this.x -= v;
      this.y -= v;
    }
    return this;
  }

  mul(v) {
    if (v instanceof Vector2) {
      this.x *= v.x;
      this.y *= v.y;
    } else {
      this.x *= v;
      this.y *= v;
    }
    return this;
  }

  div(v) {
    if (v instanceof Vector2) {
      if (v.x != 0) this.x /= v.x;
      if (v.y != 0) this.y /= v.y;
    } else {
      if (v != 0) {
        this.x /= v;
        this.y /= v;
      }
    }
    return this;
  }

  equ(v) {
    return this.x == v.x && this.y == v.y;
  }

  dot(v) {
    return this.x * v.x + this.y * v.y;
  }

  cross(v) {
    return this.x * v.y - this.y * v.x
  }

  length() {
    return Math.sqrt(this.dot(this));
  }

  norm() {
    return this.div(this.length());
  }

  static negative(v) {
    return new Vector2(-v.x, -v.y);
  }

  static add(v, b) {
    if (b instanceof Vector2) return new Vector2(v.x + b.x, v.y + b.y);
    else return new Vector2(v.x + b, v.y + b);
  }

  static sub(v, b) {
    if (b instanceof Vector2) return new Vector2(v.x - b.x, v.y - b.y);
    else return new Vector2(v.x - b, v.y - b);
  }

  static mul(v, b) {
    if (b instanceof Vector2) return new Vector2(v.x * b.x, v.y * b.y);
    else return new Vector2(v.x * b, v.y * b);
  }

  static div(v, b) {
    if (b instanceof Vector2) return new Vector2(v.x / b.x, v.y / b.y);
    else return new Vector2(v.x / b, v.y / b);
  }

  static equ(v1, v2) {
    return v1.x == v2.x && v1.y == v2.y;
  }

  static dot(v1, v2) {
    return v1.x * v2.x + v1.y * v2.y;
  }

  static cross(v1, v2) {
    return v1.x * v2.y - v1.y * v2.x;
  }

  static dist(v1, v2) {
    return Vector2.sub(v1, v2).length();
  }

  static norm(v) {
    return Vector2.div(v, v.length());
  }

}
