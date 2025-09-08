import geni.portal as portal
import geni.rspec.pg as pg

pc = portal.Context()
request = pc.makeRequestRSpec()

UBUNTU22 = "urn:publicid:IDN+emulab.net+image+GAEA:testDora.node-0"

router = request.RawPC("router")
router.disk_image = UBUNTU22
router.addService(pg.Execute(shell="bash", command="sudo sysctl -w net.ipv4.ip_forward=1"))

# --- node-0  router (10.10.0.0/24)
node0 = request.RawPC("node-0"); node0.disk_image = UBUNTU22
if0 = node0.addInterface("if0"); if0.addAddress(pg.IPv4Address("10.10.0.2", "255.255.255.0"))
r0 = router.addInterface("r0"); r0.addAddress(pg.IPv4Address("10.10.0.1", "255.255.255.0"))
link0 = request.Link("link-0"); link0.addInterface(if0); link0.addInterface(r0)
# node0.addService(pg.Execute(shell="bash", command="sudo ip route add 10.0.0.0/8 via 10.10.0.1 || true"))

# --- node-1  router (10.11.0.0/24)
node1 = request.RawPC("node-1"); node1.disk_image = UBUNTU22
if1 = node1.addInterface("if1"); if1.addAddress(pg.IPv4Address("10.11.0.2", "255.255.255.0"))
r1 = router.addInterface("r1"); r1.addAddress(pg.IPv4Address("10.11.0.1", "255.255.255.0"))
link1 = request.Link("link-1"); link1.addInterface(if1); link1.addInterface(r1)
# node1.addService(pg.Execute(shell="bash", command="sudo ip route add 10.0.0.0/8 via 10.11.0.1 || true"))

# --- node-2  router (10.12.0.0/24)
node2 = request.RawPC("node-2"); node2.disk_image = UBUNTU22
if2 = node2.addInterface("if2"); if2.addAddress(pg.IPv4Address("10.12.0.2", "255.255.255.0"))
r2 = router.addInterface("r2"); r2.addAddress(pg.IPv4Address("10.12.0.1", "255.255.255.0"))
link2 = request.Link("link-2"); link2.addInterface(if2); link2.addInterface(r2)
# node2.addService(pg.Execute(shell="bash", command="sudo ip route add 10.0.0.0/8 via 10.12.0.1 || true"))

# --- node-3  router (10.13.0.0/24)
node3 = request.RawPC("node-3"); node3.disk_image = UBUNTU22
if3 = node3.addInterface("if3"); if3.addAddress(pg.IPv4Address("10.13.0.2", "255.255.255.0"))
r3 = router.addInterface("r3"); r3.addAddress(pg.IPv4Address("10.13.0.1", "255.255.255.0"))
link3 = request.Link("link-3"); link3.addInterface(if3); link3.addInterface(r3)
# node3.addService(pg.Execute(shell="bash", command="sudo ip route add 10.0.0.0/8 via 10.13.0.1 || true"))

# --- node-4  router (10.14.0.0/24)
node4 = request.RawPC("node-4"); node4.disk_image = UBUNTU22
if4 = node4.addInterface("if4"); if4.addAddress(pg.IPv4Address("10.14.0.2", "255.255.255.0"))
r4 = router.addInterface("r4"); r4.addAddress(pg.IPv4Address("10.14.0.1", "255.255.255.0"))
link4 = request.Link("link-4"); link4.addInterface(if4); link4.addInterface(r4)
# node4.addService(pg.Execute(shell="bash", command="sudo ip route add 10.0.0.0/8 via 10.14.0.1 || true"))

# --- node-5  router (10.15.0.0/24)
node5 = request.RawPC("node-5"); node5.disk_image = UBUNTU22
if5 = node5.addInterface("if5"); if5.addAddress(pg.IPv4Address("10.15.0.2", "255.255.255.0"))
r5 = router.addInterface("r5"); r5.addAddress(pg.IPv4Address("10.15.0.1", "255.255.255.0"))
link5 = request.Link("link-5"); link5.addInterface(if5); link5.addInterface(r5)
# node5.addService(pg.Execute(shell="bash", command="sudo ip route add 10.0.0.0/8 via 10.15.0.1 || true"))

# --- node-6  router (10.16.0.0/24)
node6 = request.RawPC("node-6"); node6.disk_image = UBUNTU22
if6 = node6.addInterface("if6"); if6.addAddress(pg.IPv4Address("10.16.0.2", "255.255.255.0"))
r6 = router.addInterface("r6"); r6.addAddress(pg.IPv4Address("10.16.0.1", "255.255.255.0"))
link6 = request.Link("link-6"); link6.addInterface(if6); link6.addInterface(r6)
# node6.addService(pg.Execute(shell="bash", command="sudo ip route add 10.0.0.0/8 via 10.16.0.1 || true"))

# --- node-7  router (10.17.0.0/24)
node7 = request.RawPC("node-7"); node7.disk_image = UBUNTU22
if7 = node7.addInterface("if7"); if7.addAddress(pg.IPv4Address("10.17.0.2", "255.255.255.0"))
r7 = router.addInterface("r7"); r7.addAddress(pg.IPv4Address("10.17.0.1", "255.255.255.0"))
link7 = request.Link("link-7"); link7.addInterface(if7); link7.addInterface(r7)
# node7.addService(pg.Execute(shell="bash", command="sudo ip route add 10.0.0.0/8 via 10.17.0.1 || true"))

pc.printRequestRSpec(request)
