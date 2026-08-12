import unittest, numpy as np
import opaque_learning as q

class Tests(unittest.TestCase):
 def test_hesse(self):
  k,_,_=q.hesse();self.assertTrue(np.allclose([np.vdot(x,x) for x in k],1));self.assertTrue(np.allclose(sum(q.rho(x)/3 for x in k),np.eye(3)))
 def test_rankone_kernel(self):
  k=q.rankone_exact_kernel();self.assertTrue(np.allclose(k.sum(2),1));self.assertTrue(np.allclose(np.max(k,2),1/3))
 def test_group(self):
  p=q.infer_perms(q.rankone_exact_kernel());g=q.group(list(p));self.assertEqual(len(g),9);self.assertTrue(all(np.array_equal(q.compose(a,b),q.compose(b,a)) for a in p for b in p))
 def test_state_hitting(self):
  k=q.rankone_exact_kernel();p=q.infer_perms(k);d=q.graph_distance(list(p));v=q.bellman(k);self.assertTrue(np.allclose(np.diag(v),0));self.assertTrue(np.allclose(v[d==1],4));self.assertTrue(np.allclose(v[d==2],5))
 def test_report_again_is_nonmetric(self):
  k=q.rankone_exact_kernel();p=q.infer_perms(k);d=q.graph_distance(list(p));v=q.report_again_bellman(k);self.assertTrue(np.allclose(np.diag(v),4));self.assertTrue(np.allclose(v[d==1],4));self.assertTrue(np.allclose(v[d==2],5))
 def test_higher_rank_retains_memory(self):
  kets,_,_=q.hesse();_,kraus=q.effects_kraus(.55);branches=[]
  for state in (q.rho(kets[0]),q.rho(kets[1])):
   branch=kraus[2]@state@kraus[2];branches.append(branch/np.trace(branch))
  self.assertGreater(np.linalg.norm(branches[0]-branches[1]),.1)
 def test_null_no_information(self):
  m=q.models()[3];state=np.eye(3)/3;self.assertTrue(np.allclose(m.oracle(state,0),np.ones(9)/9))
 def test_token_shuffle_gauge(self):
  k=q.rankone_exact_kernel();perm=np.array([4,2,8,0,7,1,3,6,5]);kp=k[:,np.argsort(perm)][:,:,np.argsort(perm)];self.assertEqual(len(q.group(list(q.infer_perms(kp)))),9)
 def test_external_register_is_real_z3_square(self):
  p=q.EXTERNAL_TRANSITIONS;self.assertEqual(len(q.group(list(p))),9);self.assertTrue(all(np.array_equal(q.compose(a,b),q.compose(b,a)) for a in p for b in p))
 def test_external_register_exposure_changes_interface(self):
  m=q.models()[-1];rng=np.random.default_rng(8);seq=q.generate(m,2000,7,rng,np.arange(9),np.arange(5));token=q.infer_perms(q.kernel_from_sequences(seq));register=q.infer_perms(q.external_register_kernel(seq));self.assertFalse(all(len(np.unique(p))==9 for p in token));self.assertTrue(all(len(np.unique(p))==9 for p in register));self.assertEqual(len(q.group(list(register))),9)

if __name__=='__main__':unittest.main()
