基于langGraph开发agent，实现plan-execute-update的效果！agent中的LLM有自我排查bug和反思的能力！

![1757776032732](image/readme/1757776032732.png)

但爬虫的效果还需提升：

2025-09-14 10:58:03,560 - nodes - INFO - 当前执行 STEP: {'title': '确定数据源', 'description': '选择一个或多个提供加密货币新闻的网站，例如CoinDesk、CoinTelegraph等，确保这些网站允许爬虫访问。', 'status': 'pending'}
2025-09-14 10:58:29,797 - nodes - INFO - Tool calls detected: ["crawl_web3_news: {'urls': ['https://cointelegraph.com', 'https://cryptoslate.com', 'https://www.coindesk.com', 'https://techflowpost.com', 'https://podcasts.apple.com', 'https://www.coinglass.com/zh/news'], 'output_file': 'crypto_news.json'}"]

DevTools listening on ws://127.0.0.1:57865/devtools/browser/1df25bd2-7c8d-4d34-a578-192a4e2aefbf
[15168:23700:0914/105841.579:ERROR:google_apis\gcm\engine\registration_request.cc:291] Registration response error message: DEPRECATED_ENDPOINT
Created TensorFlow Lite XNNPACK delegate for CPU.
[23092:5020:0914/105848.455:ERROR:net\socket\ssl_client_socket_impl.cc:915] handshake failed; returned -1, SSL error code 1, net_error -100
[23092:5020:0914/105848.456:ERROR:net\socket\ssl_client_socket_impl.cc:915] handshake failed; returned -1, SSL error code 1, net_error -100
[23092:5020:0914/105848.529:ERROR:net\socket\ssl_client_socket_impl.cc:915] handshake failed; returned -1, SSL error code 1, net_error -100
[23092:5020:0914/105848.549:ERROR:net\socket\ssl_client_socket_impl.cc:915] handshake failed; returned -1, SSL error code 1, net_error -100
[23092:5020:0914/105848.631:ERROR:net\socket\ssl_client_socket_impl.cc:915] handshake failed; returned -1, SSL error code 1, net_error -100
[23092:5020:0914/105848.886:ERROR:net\socket\ssl_client_socket_impl.cc:915] handshake failed; returned -1, SSL error code 1, net_error -100
[23092:5020:0914/105901.167:ERROR:net\socket\ssl_client_socket_impl.cc:915] handshake failed; returned -1, SSL error code 1, net_error -100
[15168:23700:0914/105908.744:ERROR:google_apis\gcm\engine\connection_factory_impl.cc:483] ConnectionHandler failed with net error: -2
[23092:5020:0914/105908.747:ERROR:net\socket\ssl_client_socket_impl.cc:915] handshake failed; returned -1, SSL error code 1, net_error -100
[15168:23700:0914/105909.044:ERROR:google_apis\gcm\engine\registration_request.cc:291] Registration response error message: QUOTA_EXCEEDED
[23092:5020:0914/105910.528:ERROR:net\socket\ssl_client_socket_impl.cc:915] handshake failed; returned -1, SSL error code 1, net_error -100
[23092:5020:0914/105911.619:ERROR:net\socket\ssl_client_socket_impl.cc:915] handshake failed; returned -1, SSL error code 1, net_error -100
2025-09-14 11:26:21,177 - nodes - INFO - tool_name:crawl_web3_news, tool_args:{'urls': ['https://cointelegraph.com', 'https://cryptoslate.com', 'https://www.coindesk.com', 'https://techflowpost.com', 'https://podcasts.apple.com', 'https://www.coinglass.com/zh/news'], 'output_file': 'crypto_news.json'}
