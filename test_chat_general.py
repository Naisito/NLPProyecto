import httpx

# Crear sesión
r1 = httpx.post('http://localhost:8001/sessions')
sid = r1.json()['session_id']
print(f'✓ Sesión creada: {sid}\n')

# Test 1: Pregunta simple
r2 = httpx.post(
    f'http://localhost:8001/sessions/{sid}/chat_general',
    json={'query': '¿Cuál es la capital de Francia?'},
    timeout=30
)
print(f'✓ Chat General Test 1:')
print(f'  Status: {r2.status_code}')
print(f'  Respuesta: {r2.json()["answer"]}\n')

# Test 2: Otra pregunta
r3 = httpx.post(
    f'http://localhost:8001/sessions/{sid}/chat_general',
    json={'query': 'Cuéntame una anécdota sobre Ada Lovelace'},
    timeout=30
)
print(f'✓ Chat General Test 2:')
print(f'  Status: {r3.status_code}')
print(f'  Respuesta: {r3.json()["answer"]}\n')

print('✅ Chat sin documentos funciona correctamente!')
