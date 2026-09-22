/**
 * Planetary System 3D Simulation Canvas for ExoDip.
 * Realistic matte volumetric planets, stationary radiant central star,
 * tilted elliptical orbits, rotating asteroid belt, and starfield.
 */
(function() {
    const canvas = document.getElementById('c');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function resize() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width > 0 ? rect.width : (canvas.parentElement ? canvas.parentElement.clientWidth : 640);
        canvas.height = rect.height > 0 ? rect.height : (canvas.width < 520 ? 250 : (canvas.width < 768 ? 310 : 420));
    }
    resize();
    window.addEventListener('resize', resize);

    // Background stars
    const stars = [];
    const starPalette = ['#F1F5F9', '#F4C95D', '#7DD3C7', '#6FA8DC', '#94A3B8'];
    for (let i = 0; i < 190; i++) {
        stars.push({
            x: Math.random() * 850,
            y: Math.random() * 420,
            r: Math.random() * 1.1 + 0.2,
            color: starPalette[Math.floor(Math.random() * starPalette.length)],
            baseAlpha: Math.random() * 0.70 + 0.15,
            speed: Math.random() * 0.02 + 0.005,
            phase: Math.random() * Math.PI * 2
        });
    }

    // 3D INCLINED PERSPECTIVE — SCALED TO FIT ENTIRELY INSIDE THE BOX
    const TILT = -7 * Math.PI / 180;
    const cosT = Math.cos(TILT);
    const sinT = Math.sin(TILT);

    // Scaled orbits: fully contained within canvas boundaries (width ~600, height 420)
    const orbits = [
        { a: 88,  b: 33,  speed: 0.50, stroke: 'rgba(244, 201, 93, 0.45)', width: 1.2 }, // Orbit 1: Inner Rocky
        { a: 155, b: 58,  speed: 0.28, stroke: 'rgba(125, 211, 199, 0.42)', width: 1.2 }, // Orbit 2: Earth
        { a: 238, b: 89,  speed: 0.16, stroke: 'rgba(111, 168, 220, 0.40)', width: 1.3 }  // Orbit 3: Jupiter
    ];

    // Asteroid Belt: revolving ring between Earth and Jupiter (radius 182 to 206)
    const asteroids = [];
    for (let i = 0; i < 90; i++) {
        const angle = Math.random() * Math.PI * 2;
        const dist = Math.random() * 24 + 182;
        asteroids.push({
            dist: dist,
            bDist: dist * 0.375,
            angle: angle,
            speed: 0.20 + (Math.random() - 0.5) * 0.03,
            size: Math.random() * 1.0 + 0.35,
            color: Math.random() > 0.45 ? 'rgba(148, 163, 184, ' : 'rgba(244, 201, 93, ',
            alpha: Math.random() * 0.45 + 0.20
        });
    }

    // 3 Distinct Planets
    const planets = [
        {
            oi: 0,
            angle: 0.9,
            r: 6.0,
            type: 'rocky'
        },
        {
            oi: 1,
            angle: 2.7,
            r: 10.5,
            type: 'earth',
            rotation: 0
        },
        {
            oi: 2,
            angle: 5.1,
            r: 15.0,
            type: 'jupiter'
        }
    ];

    let lastTime = null;

    function render(now) {
        if (!lastTime) lastTime = now;
        const dt = Math.min((now - lastTime) / 1000, 0.05);
        lastTime = now;

        const W = canvas.width;
        const H = canvas.height;

        // FOCAL CENTER: Positioned in the center so the entire system fits completely inside the box
        const cx = W * 0.50;
        const cy = H * 0.50;
        const scale = Math.min(1.0, Math.max(0.48, W / 580));

        // 1. Ultra-Dark Space Background with warm ambient stellar radiance
        const bg = ctx.createRadialGradient(cx, cy, 25 * scale, cx, cy, Math.max(W, H) * 0.85);
        bg.addColorStop(0, 'rgba(36, 28, 14, 0.48)');     // Subtle warm golden solar ambiance
        bg.addColorStop(0.30, 'rgba(14, 21, 36, 0.85)');  // Main surface
        bg.addColorStop(0.70, 'rgba(7, 11, 20, 0.98)');   // Deep background
        bg.addColorStop(1, '#070B14');                     // Deep space canvas background
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, W, H);

        // Subtle galactic star dust wash across the background
        ctx.save();
        ctx.rotate(-0.25);
        const dustGrad = ctx.createLinearGradient(0, -H, W * 1.4, H * 1.4);
        dustGrad.addColorStop(0, 'transparent');
        dustGrad.addColorStop(0.48, 'rgba(111, 168, 220, 0.025)');
        dustGrad.addColorStop(0.52, 'rgba(244, 201, 93, 0.020)');
        dustGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = dustGrad;
        ctx.fillRect(-W * 0.5, -H * 0.5, W * 2, H * 2);
        ctx.restore();

        // 2. Stars
        stars.forEach(s => {
            const alpha = s.baseAlpha + Math.sin(now * s.speed + s.phase) * 0.15;
            ctx.beginPath();
            ctx.arc(s.x * (W / 850), s.y * (H / 420), Math.max(0.3, s.r * scale), 0, Math.PI * 2);
            ctx.fillStyle = s.color;
            ctx.globalAlpha = Math.max(0.10, Math.min(1.0, alpha));
            ctx.fill();
        });
        ctx.globalAlpha = 1.0;

        // 3. Complete, Unclipped 3D Perspective Orbit Rings
        orbits.forEach(orb => {
            ctx.beginPath();
            const steps = 110;
            for (let i = 0; i <= steps; i++) {
                const theta = (i / steps) * Math.PI * 2;
                const ox = (orb.a * scale) * Math.cos(theta);
                const oy = (orb.b * scale) * Math.sin(theta);
                const rx = ox * cosT - oy * sinT;
                const ry = ox * sinT + oy * cosT;
                const px = cx + rx;
                const py = cy + ry;

                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.closePath();

            ctx.strokeStyle = orb.stroke;
            ctx.lineWidth = Math.max(0.75, orb.width * scale);
            ctx.stroke();
        });

        // 4. Asteroid Belt Particles
        asteroids.forEach(a => {
            a.angle = (a.angle + a.speed * dt) % (Math.PI * 2);
            const ax = (a.dist * scale) * Math.cos(a.angle);
            const ay = (a.bDist * scale) * Math.sin(a.angle);
            const rx = ax * cosT - ay * sinT;
            const ry = ax * sinT + ay * cosT;
            const px = cx + rx;
            const py = cy + ry;

            ctx.beginPath();
            ctx.arc(px, py, Math.max(0.3, a.size * scale), 0, Math.PI * 2);
            ctx.fillStyle = a.color + a.alpha + ')';
            ctx.fill();
        });

        // 5. Gather Render Queue for Depth Sorting
        const renderQueue = [];

        // Central Star at (cx, cy)
        renderQueue.push({
            type: 'star',
            depth: 0,
            x: cx,
            y: cy
        });

        // Planets
        planets.forEach(pl => {
            const orb = orbits[pl.oi];
            pl.angle = (pl.angle + orb.speed * dt) % (Math.PI * 2);
            if (pl.type === 'earth') pl.rotation += 0.35 * dt;

            const ox = (orb.a * scale) * Math.cos(pl.angle);
            const oy = (orb.b * scale) * Math.sin(pl.angle);
            const rx = ox * cosT - oy * sinT;
            const ry = ox * sinT + oy * cosT;
            const px = cx + rx;
            const py = cy + ry;

            // Depth based on sine (negative = behind star, positive = in front)
            const depth = Math.sin(pl.angle);

            renderQueue.push({
                type: 'planet',
                planet: pl,
                depth: depth,
                x: px,
                y: py,
                r: Math.max(3.5, pl.r * Math.max(0.65, scale))
            });
        });

        // Sort by depth (farthest behind rendered first, closest foreground rendered last)
        renderQueue.sort((a, b) => a.depth - b.depth);

        // 6. Draw in Depth Order
        renderQueue.forEach(item => {
            if (item.type === 'star') {
                drawStar(item.x, item.y, scale);
            } else {
                drawPlanet(item, cx, cy);
            }
        });

        requestAnimationFrame(render);
    }

    // === COMPLETE, UNCLIPPED CENTRAL STAR ===
    function drawStar(sx, sy, scale = 1.0) {
        const starR = Math.max(16, 30 * scale);

        // Soft, warm coronal halo radiating into space
        const corona = ctx.createRadialGradient(sx, sy, starR * 0.7, sx, sy, starR * 2.4);
        corona.addColorStop(0, 'rgba(244, 201, 93, 0.85)');
        corona.addColorStop(0.30, 'rgba(244, 201, 93, 0.38)');
        corona.addColorStop(0.70, 'rgba(244, 201, 93, 0.10)');
        corona.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.arc(sx, sy, starR * 2.4, 0, Math.PI * 2);
        ctx.fillStyle = corona;
        ctx.fill();

        // Single cohesive gold star body (clean, warm, radiant #F4C95D)
        const bodyGrad = ctx.createRadialGradient(sx, sy, 0, sx, sy, starR);
        bodyGrad.addColorStop(0, '#FFFDF5');
        bodyGrad.addColorStop(0.35, '#FCE8A6');
        bodyGrad.addColorStop(0.72, '#F4C95D');
        bodyGrad.addColorStop(1.0, '#D4A83E');
        ctx.beginPath();
        ctx.arc(sx, sy, starR, 0, Math.PI * 2);
        ctx.fillStyle = bodyGrad;
        ctx.fill();

        // Delicate line outwards of its surface (clean, fine single outer ring)
        ctx.beginPath();
        ctx.arc(sx, sy, starR + 5, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(244, 201, 93, 0.75)';
        ctx.lineWidth = 1.1;
        ctx.stroke();
    }

    // === VOLUMETRIC REALISTIC PLANETS (MATTE FINISH) ===
    function drawPlanet(item, starX, starY) {
        const { x, y, r, planet } = item;

        // Vector pointing from planet toward the central star (reduced offset for matte finish)
        const angleToStar = Math.atan2(starY - y, starX - x);
        const lightDx = Math.cos(angleToStar) * (r * 0.28);
        const lightDy = Math.sin(angleToStar) * (r * 0.28);

        ctx.save();
        ctx.translate(x, y);

        if (planet.type === 'earth') {
            // 1. CRYSTAL-CLEAR AZURE / OCEAN WORLD (WITH SUBTLE SHINE LIKE PLANET 3)
            ctx.save();
            ctx.beginPath();
            ctx.arc(0, 0, r, 0, Math.PI * 2);
            ctx.clip(); // Mask to sphere

            // Base atmospheric gradient with gentle sunlit sheen
            const azureGrad = ctx.createRadialGradient(lightDx, lightDy, r * 0.20, 0, 0, r);
            azureGrad.addColorStop(0, '#F1F5F9');    // Main text highlight sheen
            azureGrad.addColorStop(0.30, '#7DD3C7'); // Accent cyan
            azureGrad.addColorStop(0.65, '#6FA8DC'); // Primary blue body
            azureGrad.addColorStop(1.0, '#070B14');  // Night side shadow
            ctx.fillStyle = azureGrad;
            ctx.fill();

            // Clear horizontal atmospheric / ocean current belts
            ctx.fillStyle = 'rgba(111, 168, 220, 0.36)'; // Primary blue zone
            ctx.fillRect(-r, -r * 0.38, r * 2, r * 0.16);
            ctx.fillRect(-r, r * 0.08, r * 2, r * 0.20);
            ctx.fillRect(-r, r * 0.48, r * 2, r * 0.12);

            ctx.fillStyle = 'rgba(241, 245, 249, 0.26)'; // Bright white cloud sheen belt
            ctx.fillRect(-r, -r * 0.18, r * 2, r * 0.14);
            ctx.fillRect(-r, r * 0.30, r * 2, r * 0.12);

            // Great Azure Spot / cyclone oval
            ctx.fillStyle = 'rgba(125, 211, 199, 0.65)';
            ctx.beginPath();
            ctx.ellipse(lightDx * 0.35 + 2.0, lightDy * 0.35 + r * 0.14, r * 0.22, r * 0.12, 0.05, 0, Math.PI * 2);
            ctx.fill();

            // Night-side volumetric shadow mask (clean 3D globe effect)
            const shadowGrad = ctx.createRadialGradient(-lightDx * 0.75, -lightDy * 0.75, r * 0.15, 0, 0, r);
            shadowGrad.addColorStop(0, 'rgba(7, 11, 20, 0.98)');
            shadowGrad.addColorStop(0.55, 'rgba(14, 21, 36, 0.75)');
            shadowGrad.addColorStop(1.0, 'transparent');
            ctx.fillStyle = shadowGrad;
            ctx.fill();

            ctx.restore();

            // Delicate illuminated outer limb edge
            ctx.beginPath();
            ctx.arc(0, 0, r, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(125, 211, 199, 0.40)';
            ctx.lineWidth = 0.9;
            ctx.stroke();

        } else if (planet.type === 'rocky') {
            // 2. INNER ROCKY PLANET (Mercury/Moon-like)
            ctx.save();
            ctx.beginPath();
            ctx.arc(0, 0, r, 0, Math.PI * 2);
            ctx.clip();

            const rockyGrad = ctx.createRadialGradient(lightDx, lightDy, r * 0.35, 0, 0, r);
            rockyGrad.addColorStop(0, '#94a3b8');    // Muted sunlit basalt
            rockyGrad.addColorStop(0.42, '#64748b'); // Weathered rock
            rockyGrad.addColorStop(0.75, '#334155'); // Cratered shadow
            rockyGrad.addColorStop(1.0, '#020617');  // Deep shadow
            ctx.fillStyle = rockyGrad;
            ctx.fill();

            // Surface crater detail
            ctx.fillStyle = 'rgba(30, 41, 59, 0.45)';
            ctx.beginPath();
            ctx.arc(lightDx * 0.4 - 1, lightDy * 0.4 + 1, r * 0.22, 0, Math.PI * 2);
            ctx.fill();

            ctx.restore();

            ctx.beginPath();
            ctx.arc(0, 0, r, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.20)';
            ctx.lineWidth = 0.8;
            ctx.stroke();

        } else if (planet.type === 'jupiter') {
            // 3. JUPITER-LIKE BANDED GAS GIANT
            ctx.save();
            ctx.beginPath();
            ctx.arc(0, 0, r, 0, Math.PI * 2);
            ctx.clip();

            // Base planetary gas gradient (matte, subdued)
            const jupGrad = ctx.createRadialGradient(lightDx, lightDy, r * 0.38, 0, 0, r);
            jupGrad.addColorStop(0, '#b8924a');    // Muted warm tan
            jupGrad.addColorStop(0.40, '#9a7040'); // Amber atmospheric zone
            jupGrad.addColorStop(0.75, '#6b4423'); // Deep brown belt shadow
            jupGrad.addColorStop(1.0, '#1c0f05');  // Night side shadow
            ctx.fillStyle = jupGrad;
            ctx.fill();

            // Horizontal atmospheric belts
            ctx.fillStyle = 'rgba(107, 68, 35, 0.35)';
            ctx.fillRect(-r, -r * 0.38, r * 2, r * 0.16);
            ctx.fillRect(-r, r * 0.08, r * 2, r * 0.22);
            ctx.fillRect(-r, r * 0.48, r * 2, r * 0.12);

            ctx.fillStyle = 'rgba(255, 248, 235, 0.22)';
            ctx.fillRect(-r, -r * 0.18, r * 2, r * 0.14);
            ctx.fillRect(-r, r * 0.32, r * 2, r * 0.12);

            // Great Red Spot / storm oval
            ctx.fillStyle = 'rgba(194, 65, 12, 0.65)';
            ctx.beginPath();
            ctx.ellipse(lightDx * 0.35 + 2.5, lightDy * 0.35 + r * 0.18, r * 0.22, r * 0.12, 0.05, 0, Math.PI * 2);
            ctx.fill();

            // Night-side shadow mask
            const shadowGrad = ctx.createRadialGradient(-lightDx * 0.75, -lightDy * 0.75, r * 0.15, 0, 0, r);
            shadowGrad.addColorStop(0, 'rgba(4, 3, 2, 0.98)');
            shadowGrad.addColorStop(0.55, 'rgba(4, 3, 2, 0.75)');
            shadowGrad.addColorStop(1.0, 'transparent');
            ctx.fillStyle = shadowGrad;
            ctx.fill();

            ctx.restore();

            // Delicate limb edge
            ctx.beginPath();
            ctx.arc(0, 0, r, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(229, 213, 181, 0.30)';
            ctx.lineWidth = 0.9;
            ctx.stroke();
        }

        ctx.restore();
    }

    requestAnimationFrame(render);
})();
