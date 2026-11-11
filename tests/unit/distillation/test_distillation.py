"""
Unit tests for Kilat Knowledge Distillation: Losses, Student & Teacher Wrappers.
Follows BDD standards from UNIT_TESTING_STRUCTURE.md.
"""

import pytest
import torch
import torch.nn as nn

from kilat.distillation.losses import VanillaKDLoss, ReverseKDLoss, AdaptiveKDLoss, DistillLossOutput
from kilat.distillation.student import BaseStudent
from kilat.distillation.teacher import BaseTeacher
from tests.shared.bdd import Scenario, Given, When, Then, And


class MockStudent(BaseStudent):
    def __init__(self, vocab: int = 100):
        super().__init__()
        self._vocab = vocab
        self.linear = nn.Linear(32, vocab)

    @classmethod
    def load(cls, *args, **kwargs):
        return cls()

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch, seq_len = input_ids.shape
        return torch.randn(batch, seq_len, self._vocab)

    @property
    def vocab_size(self):
        return self._vocab


class MockTeacher(BaseTeacher):
    def __init__(self, vocab: int = 100):
        super().__init__()
        self._vocab = vocab
        self.fc = nn.Linear(32, vocab)
        self._freeze()

    @classmethod
    def load(cls, *args, **kwargs):
        return cls()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch, seq_len = input_ids.shape
        return torch.randn(batch, seq_len, self._vocab)

    @property
    def vocab_size(self):
        return self._vocab


def test_vanilla_kd_loss():
    with Scenario("Unit: Vanilla Knowledge Distillation Loss"):
        with Given("synthetic student logits, teacher logits, and target labels"):
            B, T, V = 2, 4, 10
            student_logits = torch.randn(B, T, V, requires_grad=True)
            teacher_logits = torch.randn(B, T, V)
            labels = torch.randint(0, V, (B, T))

            with When("VanillaKDLoss is computed with temperature=2.0 and alpha=0.5"):
                kd_loss_fn = VanillaKDLoss(temperature=2.0, alpha=0.5)
                output = kd_loss_fn(student_logits, teacher_logits, labels)

                with Then("a DistillLossOutput container is returned"):
                    assert isinstance(output, DistillLossOutput)

                with And("total_loss is positive and numerically finite"):
                    assert output.total_loss.item() > 0.0
                    assert torch.isfinite(output.total_loss)

                with And("the logging dictionary contains total, kl, and ce loss values"):
                    log_data = output.log_dict()
                    assert "loss/total" in log_data
                    assert "loss/kl" in log_data
                    assert "loss/ce" in log_data


def test_reverse_and_adaptive_kd_losses():
    with Scenario("Unit: Reverse and Adaptive KD Loss Formulations"):
        with Given("logits and labels tensors"):
            B, T, V = 2, 4, 10
            student_logits = torch.randn(B, T, V, requires_grad=True)
            teacher_logits = torch.randn(B, T, V)
            labels = torch.randint(0, V, (B, T))

            with When("ReverseKDLoss is evaluated"):
                rev_loss_fn = ReverseKDLoss(temperature=2.0, alpha=0.5)
                rev_out = rev_loss_fn(student_logits, teacher_logits, labels)
                with Then("the reverse distillation loss is finite"):
                    assert torch.isfinite(rev_out.total_loss)

            with When("AdaptiveKDLoss is evaluated"):
                adapt_loss_fn = AdaptiveKDLoss()
                adapt_out = adapt_loss_fn(student_logits, teacher_logits, labels)
                with Then("the adaptive distillation loss is finite"):
                    assert torch.isfinite(adapt_out.total_loss)


def test_teacher_student_contracts():
    with Scenario("Unit: Teacher and Student Model Abstractions"):
        with Given("a MockTeacher and MockStudent"):
            teacher = MockTeacher(vocab=50)
            student = MockStudent(vocab=50)

            with When("verifying teacher frozen status"):
                with Then("all teacher parameters have requires_grad=False"):
                    assert all(not p.requires_grad for p in teacher.parameters())
                with And("calling train() on teacher returns self safely"):
                    res = teacher.train(True)
                    assert res is teacher

            with When("verifying student properties"):
                with Then("student parameter count is positive"):
                    assert student.num_parameters > 0
                with And("vocab_size matches teacher"):
                    assert student.vocab_size == teacher.vocab_size
                with And("repr functions format parameter summaries"):
                    assert "MockStudent" in repr(student)
                    assert "MockTeacher" in repr(teacher)

        with Given("bare subclasses of BaseStudent and BaseTeacher without overrides"):
            class BareStudent(BaseStudent):
                @classmethod
                def load(cls, *args, **kwargs): return cls()
                @classmethod
                def from_config(cls, *args, **kwargs): return cls()
                def forward(self, input_ids, **kwargs):
                    return input_ids

            class BareTeacher(BaseTeacher):
                @classmethod
                def load(cls, *args, **kwargs): return cls()
                def forward(self, input_ids, **kwargs):
                    return input_ids

            bare_s = BareStudent()
            bare_t = BareTeacher()

            with When("accessing default unimplemented vocab_size"):
                with Then("NotImplementedError is raised"):
                    with pytest.raises(NotImplementedError):
                        _ = bare_s.vocab_size
                    with pytest.raises(NotImplementedError):
                        _ = bare_t.vocab_size
