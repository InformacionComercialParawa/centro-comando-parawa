// Service Worker - Centro de Comando Parawa
const CACHE_NAME = 'centro-comando-v1';
const ASSETS_TO_CACHE = [
    '/',
    '/assets/manifest.json'
];

// Instalación: cachear assets básicos
self.addEventListener('install', function(event) {
    event.waitUntil(
        caches.open(CACHE_NAME).then(function(cache) {
            console.log('✅ Service Worker: Cache abierto');
            return cache.addAll(ASSETS_TO_CACHE);
        })
    );
    self.skipWaiting();
});

// Activación: limpiar caches viejos
self.addEventListener('activate', function(event) {
    event.waitUntil(
        caches.keys().then(function(cacheNames) {
            return Promise.all(
                cacheNames.filter(function(name) {
                    return name !== CACHE_NAME;
                }).map(function(name) {
                    console.log('🗑️ Service Worker: Eliminando cache viejo:', name);
                    return caches.delete(name);
                })
            );
        })
    );
    self.clients.claim();
});

// Fetch: network-first con fallback a cache
self.addEventListener('fetch', function(event) {
    event.respondWith(
        fetch(event.request)
            .then(function(response) {
                // Si la respuesta es válida, clonarla y cachearla
                if (response && response.status === 200 && response.type === 'basic') {
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then(function(cache) {
                        cache.put(event.request, responseClone);
                    });
                }
                return response;
            })
            .catch(function() {
                // Sin red: intentar desde cache
                return caches.match(event.request).then(function(cached) {
                    if (cached) {
                        return cached;
                    }
                    // Fallback básico para navegación offline
                    if (event.request.mode === 'navigate') {
                        return caches.match('/');
                    }
                });
            })
    );
});
