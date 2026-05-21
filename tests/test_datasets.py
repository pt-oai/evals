from __future__ import annotations

import json

import pytest

from prism_evals.datasets import dataset_sha256, load_dataset


def test_loads_one_yaml_item_per_file_in_stable_order(tmp_path):
    dataset = tmp_path / "scenarios"
    dataset.mkdir()
    (dataset / "b.yaml").write_text(
        """
id: b-case
tags: [support]
turns:
  - user: hello
  - assistant_seed: hi there
  - user: follow up
""",
        encoding="utf-8",
    )
    (dataset / "a.json").write_text(
        json.dumps({"id": "a-case", "turns": [{"user": "start"}]}),
        encoding="utf-8",
    )
    (dataset / "_shared.yaml").write_text("id: ignored\n", encoding="utf-8")

    items = load_dataset(dataset)

    assert [item.item_id for item in items] == ["a-case", "b-case"]
    assert items[0].source_path == "a.json"
    assert items[1].data["turns"][0] == {"id": "turn_01", "role": "user", "content": "hello"}
    assert items[1].data["turns"][1] == {
        "id": "turn_02",
        "role": "assistant",
        "mode": "seed",
        "content": "hi there",
    }
    assert len(dataset_sha256(dataset)) == 64


def test_duplicate_folder_item_ids_fail(tmp_path):
    dataset = tmp_path / "scenarios"
    dataset.mkdir()
    (dataset / "a.yaml").write_text("id: same\nturns: []\n", encoding="utf-8")
    (dataset / "b.yaml").write_text("id: same\nturns: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate dataset item id"):
        load_dataset(dataset)


def test_csv_can_reference_scenario_file(tmp_path):
    scenario = tmp_path / "case.yaml"
    scenario.write_text("id: from-file\nturns:\n  - user: hello\n", encoding="utf-8")
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,scenario_path,tags\ncsv-id,case.yaml,\"a,b\"\n", encoding="utf-8")

    items = load_dataset(csv_path)

    assert items[0].item_id == "csv-id"
    assert items[0].data["id"] == "from-file"
    assert items[0].data["tags"] == ["a", "b"]
    assert items[0].data["turns"][0]["content"] == "hello"


def test_csv_turns_json_is_expanded(tmp_path):
    csv_path = tmp_path / "data.csv"
    turns_json = json.dumps([{"user": "hello"}, {"assistant_expect": {"contains": "hi"}}])
    csv_path.write_text(f'id,turns_json\ncase-1,"{turns_json.replace(chr(34), chr(34) * 2)}"\n', encoding="utf-8")

    items = load_dataset(csv_path)

    assert items[0].data["turns"][0]["role"] == "user"
    assert items[0].data["turns"][1]["role"] == "assistant"
    assert items[0].data["turns"][1]["mode"] == "expect"
