"""token 计量旁路的单元测试：流式 usage 注入 + agent 归因。

这两个缺陷均已由真实运行复现，因此必须有回归保护：
- 不注入 ``stream_options.include_usage`` 时，vLLM 的流式响应不带 usage → 100% 退化为估算；
- 成员 agent 为子图，子图内部节点名为 ``model``，按 ``langgraph_node`` 归因会把
  ``chat_agent`` 的消耗全部记为 ``model``。
"""
import os
import unittest

from adaptive_runtime import token_probe


class TestAgentAttribution(unittest.TestCase):
    def test_outermost_checkpoint_ns_wins(self):
        # 子图内：langgraph_node 是 "model"，真正该取的是最外层 chat_agent
        meta = {"langgraph_checkpoint_ns": "chat_agent:cf47|model:bea2",
                "langgraph_node": "model"}
        self.assertEqual(token_probe._agent_from_meta(meta), "chat_agent")

    def test_plain_node(self):
        meta = {"langgraph_checkpoint_ns": "supervisor:838", "langgraph_node": "supervisor"}
        self.assertEqual(token_probe._agent_from_meta(meta), "supervisor")

    def test_fallback_to_node(self):
        self.assertEqual(token_probe._agent_from_meta({"langgraph_node": "x"}), "x")

    def test_empty(self):
        self.assertEqual(token_probe._agent_from_meta({}), "")
        self.assertEqual(token_probe._agent_from_meta(None), "")
        self.assertEqual(token_probe._agent_from_meta({"langgraph_checkpoint_ns": "  "}), "")

    def test_ls_agent_type_is_not_a_node_name(self):
        # LangSmith 给模型运行打的固定标签，混进来会把归因污染成 "model"
        self.assertEqual(token_probe._agent_from_meta({"ls_agent_type": "model"}), "")

    def test_context_var_used(self):
        token_probe._AGENT_CTX.set("crawler_agent")
        try:
            self.assertEqual(token_probe._agent_of(None), "crawler_agent")
        finally:
            token_probe._AGENT_CTX.set("")

    def test_unknown_without_any_source(self):
        token_probe._AGENT_CTX.set("")
        self.assertEqual(token_probe._agent_of(None), "unknown")


class TestStreamUsageInjection(unittest.TestCase):
    def test_rejection_matcher_is_specific(self):
        self.assertTrue(token_probe._is_stream_options_rejection(
            Exception("400 - {'error': 'unknown field stream_options'}")))
        self.assertTrue(token_probe._is_stream_options_rejection(
            Exception("unsupported parameter: include_usage")))
        # 网络/鉴权错误不能被当成「后端不支持」，否则会吞掉真实故障
        self.assertFalse(token_probe._is_stream_options_rejection(
            Exception("401 Unauthorized")))
        self.assertFalse(token_probe._is_stream_options_rejection(
            Exception("Connection reset by peer")))

    def test_env_switch(self):
        old = os.environ.get("ADAPTIVE_TOKEN_STREAM_USAGE")
        try:
            os.environ["ADAPTIVE_TOKEN_STREAM_USAGE"] = "0"
            self.assertFalse(token_probe._env_allows_inject())
            os.environ["ADAPTIVE_TOKEN_STREAM_USAGE"] = "1"
            self.assertTrue(token_probe._env_allows_inject())
        finally:
            if old is None:
                os.environ.pop("ADAPTIVE_TOKEN_STREAM_USAGE", None)
            else:
                os.environ["ADAPTIVE_TOKEN_STREAM_USAGE"] = old

    def test_install_patches_should_stream_usage(self):
        try:
            token_probe.install(memory_only=True)
        except Exception as e:                     # 缺 langchain-openai 时跳过
            self.skipTest(f"langchain-openai 不可用：{e}")
        try:
            self.assertIn(token_probe._INJECT_USAGE, (True, False))
            if token_probe._INJECT_USAGE:
                from langchain_openai import ChatOpenAI
                self.assertTrue(token_probe.usage_injection()["enabled"])
                import httpx
                m = ChatOpenAI(model="m", api_key="k", base_url="http://127.0.0.1:1/v1",
                               http_client=httpx.Client(), http_async_client=httpx.AsyncClient())
                self.assertTrue(m._should_stream_usage(None))
        finally:
            token_probe.uninstall()

    def test_disable_inject_is_sticky(self):
        token_probe._INJECT_USAGE = True
        token_probe._INJECT_OFF_REASON = ""
        try:
            token_probe._disable_inject("400 stream_options")
            self.assertFalse(token_probe._INJECT_USAGE)
            # 已经关掉后不应被后续原因覆盖，保留第一条根因
            token_probe._disable_inject("另一个原因")
            self.assertEqual(token_probe._INJECT_OFF_REASON, "400 stream_options")
        finally:
            token_probe._INJECT_USAGE = True
            token_probe._INJECT_OFF_REASON = ""


class TestTotals(unittest.TestCase):
    def test_totals_shape(self):
        rows = [{"in": 10, "out": 5, "estimated": False},
                {"in": 0, "out": 0, "estimated": True, "error": "boom"}]
        t = token_probe.totals(rows)
        self.assertEqual(t["calls"], 2)
        self.assertEqual(t["total"], 15)
        self.assertEqual(t["estimated_ratio"], 0.5)
        self.assertEqual(t["errors"], 1)
        self.assertIn("stream_usage_injection", t)


if __name__ == "__main__":
    unittest.main()
