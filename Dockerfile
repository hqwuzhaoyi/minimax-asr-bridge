# syntax=docker/dockerfile:1
FROM rust:1-slim AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM gcr.io/distroless/cc-debian12
COPY --from=build /src/target/release/minimax-asr-bridge /usr/local/bin/minimax-asr-bridge
ENV LISTEN_ADDR=0.0.0.0:8090
EXPOSE 8090
USER nonroot:nonroot
ENTRYPOINT ["/usr/local/bin/minimax-asr-bridge"]
