# Stage 1: Build stage
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .

# Nhận biến VITE_GRADIO_URL khi build image để Vite nhúng vào build bundle
ARG VITE_GRADIO_URL
ENV VITE_GRADIO_URL=$VITE_GRADIO_URL
RUN npm run build

# Stage 2: Production stage sử dụng Nginx
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
# Sao chép cấu hình Nginx tự chọn nếu cần routing SPA
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
