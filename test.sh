export ANTHROPIC_MODEL=claude-sonnet-4-20250514
export ANTHROPIC_API_KEY="sk-fc8ccc5aa0025dd0816f3d57a6d285ca307bafc80107a3aa52eeeab82b8c7bab"
echo $ANTHROPIC_API_KEY

# 测试调用
curl -v -x http://127.0.0.1:5160 \
	-H "x-api-key: $ANTHROPIC_API_KEY" \
	-H "anthropic-version: 2023-06-01" \
	-H "content-type: application/json" \
	-d '{
           "model": "claude-sonnet-4-20250514",
            "max_tokens": 10,
	    "messages": [{"role": "user", "content": "Hello"}]
	    }' \
	https://api.anthropic.com/v1/messages
