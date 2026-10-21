# futsalcut-web: web app toolchain (Node + Playwright driving Google Chrome; Chromium lacks H.264/AAC).
# Build on the test host:  docker build -f docker/web.Dockerfile -t futsalcut-web docker/
FROM node:22-bookworm-slim

ENV npm_config_update_notifier=false \
    npm_config_fund=false

# Google Chrome stable (system install under /opt/google/chrome) plus its OS dependencies.
RUN npx -y playwright@1.63.0 install --with-deps chrome \
 && rm -rf /var/lib/apt/lists/* /root/.npm /root/.cache

# Project installs must not fetch Playwright's bundled browsers; tests use channel "chrome".
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

WORKDIR /work
