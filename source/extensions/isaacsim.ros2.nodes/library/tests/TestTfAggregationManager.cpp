// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <doctest/doctest.h>
#include <isaacsim/ros2/nodes/TfAggregationManager.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace isaacsim::ros2::core;
using namespace isaacsim::ros2::nodes;

namespace
{

class FakeContextHandle : public Ros2ContextHandle
{
public:
    explicit FakeContextHandle(size_t domainId = 0) : m_domainId(domainId)
    {
    }

    void* getContext() override
    {
        return this;
    }

    void init(int argc, char const* const* argv, bool setDomainId = false, size_t domainId = 0) override
    {
        (void)argc;
        (void)argv;
        if (setDomainId)
        {
            m_domainId = domainId;
        }
        m_valid = true;
    }

    bool isValid() override
    {
        return m_valid;
    }

    bool shutdown(const char* shutdownReason = nullptr) override
    {
        (void)shutdownReason;
        m_valid = false;
        return true;
    }

    size_t getDomainId() override
    {
        return m_valid ? m_domainId : (std::numeric_limits<size_t>::max)();
    }

private:
    size_t m_domainId = 0;
    bool m_valid = true;
};

class FakeNodeHandle : public Ros2NodeHandle
{
public:
    explicit FakeNodeHandle(Ros2ContextHandle* contextHandle) : m_contextHandle(contextHandle)
    {
    }

    Ros2ContextHandle* getContextHandle() override
    {
        return m_contextHandle;
    }

    void* getNode() override
    {
        return this;
    }

private:
    Ros2ContextHandle* m_contextHandle = nullptr;
};

class FakePublisher : public Ros2Publisher
{
public:
    explicit FakePublisher(size_t subscriptionCount) : m_subscriptionCount(subscriptionCount)
    {
    }

    void publish(const void* msg) override
    {
        m_lastMessage = msg;
        m_publishCount++;
    }

    size_t getSubscriptionCount() override
    {
        return m_subscriptionCount;
    }

    bool isValid() override
    {
        return true;
    }

    size_t publishCount() const
    {
        return m_publishCount;
    }

    const void* lastMessage() const
    {
        return m_lastMessage;
    }

private:
    size_t m_subscriptionCount = 0;
    size_t m_publishCount = 0;
    const void* m_lastMessage = nullptr;
};

class FakeTfTreeMessage : public Ros2TfTreeMessage
{
public:
    FakeTfTreeMessage()
    {
        m_msg = this;
    }

    const void* getTypeSupportHandle() override
    {
        return this;
    }

    void writeData(const double& timeStamp, std::vector<TfTransformStamped>& transforms) override
    {
        lastTimestamp = timeStamp;
        lastTransforms = transforms;
        writeCount++;
    }

    void readData(std::vector<TfTransformStamped>& transforms) override
    {
        transforms = lastTransforms;
    }

    double lastTimestamp = 0.0;
    size_t writeCount = 0;
    std::vector<TfTransformStamped> lastTransforms;
};

class FakeFactory : public Ros2Factory
{
public:
    std::shared_ptr<Ros2ContextHandle> createContextHandle() override
    {
        auto context = std::make_shared<FakeContextHandle>(0);
        contexts.push_back(context);
        return context;
    }

    std::shared_ptr<Ros2NodeHandle> createNodeHandle(const char* name,
                                                     const char* namespaceName,
                                                     Ros2ContextHandle* contextHandle) override
    {
        (void)name;
        (void)namespaceName;
        auto node = std::make_shared<FakeNodeHandle>(contextHandle);
        nodeHandles.push_back(node);
        return node;
    }

    std::shared_ptr<Ros2Publisher> createPublisher(Ros2NodeHandle* nodeHandle,
                                                   const char* topicName,
                                                   const void* typeSupport,
                                                   const Ros2QoSProfile& qos) override
    {
        (void)typeSupport;
        publisherNodeHandles.push_back(nodeHandle);
        publisherTopics.emplace_back(topicName);
        publisherQos.push_back(qos);
        auto publisher = std::make_shared<FakePublisher>(nextSubscriptionCount);
        publishers.push_back(publisher);
        return publisher;
    }

    std::shared_ptr<Ros2Subscriber> createSubscriber(Ros2NodeHandle* nodeHandle,
                                                     const char* topicName,
                                                     const void* typeSupport,
                                                     const Ros2QoSProfile& qos) override
    {
        (void)nodeHandle;
        (void)topicName;
        (void)typeSupport;
        (void)qos;
        return nullptr;
    }

    std::shared_ptr<Ros2Service> createService(Ros2NodeHandle* nodeHandle,
                                               const char* serviceName,
                                               const void* typeSupport,
                                               const Ros2QoSProfile& qos) override
    {
        (void)nodeHandle;
        (void)serviceName;
        (void)typeSupport;
        (void)qos;
        return nullptr;
    }

    std::shared_ptr<Ros2Client> createClient(Ros2NodeHandle* nodeHandle,
                                             const char* serviceName,
                                             const void* typeSupport,
                                             const Ros2QoSProfile& qos) override
    {
        (void)nodeHandle;
        (void)serviceName;
        (void)typeSupport;
        (void)qos;
        return nullptr;
    }

    std::shared_ptr<Ros2ClockMessage> createClockMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2ImuMessage> createImuMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2CameraInfoMessage> createCameraInfoMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2ImageMessage> createImageMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2CompressedImageMessage> createCompressedImageMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2NitrosBridgeImageMessage> createNitrosBridgeImageMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2BoundingBox2DMessage> createBoundingBox2DMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2BoundingBox3DMessage> createBoundingBox3DMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2OdometryMessage> createOdometryMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2RawTfTreeMessage> createRawTfTreeMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2SemanticLabelMessage> createSemanticLabelMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2JointStateMessage> createJointStateMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2PointCloudMessage> createPointCloudMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2LaserScanMessage> createLaserScanMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2TfTreeMessage> createTfTreeMessage() override
    {
        auto message = std::make_shared<FakeTfTreeMessage>();
        messages.push_back(message);
        return message;
    }

    std::shared_ptr<Ros2TwistMessage> createTwistMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2AckermannDriveStampedMessage> createAckermannDriveStampedMessage() override
    {
        return nullptr;
    }

    std::shared_ptr<Ros2Message> createDynamicMessage(const std::string& pkgName,
                                                      const std::string& msgSubfolder,
                                                      const std::string& msgName,
                                                      BackendMessageType messageType) override
    {
        (void)pkgName;
        (void)msgSubfolder;
        (void)msgName;
        (void)messageType;
        return nullptr;
    }

    bool validateTopicName(const std::string& topicName) override
    {
        return !topicName.empty();
    }

    bool validateNamespaceName(const std::string& namespaceName) override
    {
        (void)namespaceName;
        return true;
    }

    bool validateNodeName(const std::string& nodeName) override
    {
        return !nodeName.empty();
    }

    size_t nextSubscriptionCount = 1;
    std::vector<Ros2NodeHandle*> publisherNodeHandles;
    std::vector<std::string> publisherTopics;
    std::vector<Ros2QoSProfile> publisherQos;
    std::vector<std::shared_ptr<FakePublisher>> publishers;
    std::vector<std::shared_ptr<FakeTfTreeMessage>> messages;
    std::vector<std::shared_ptr<FakeContextHandle>> contexts;
    std::vector<std::shared_ptr<FakeNodeHandle>> nodeHandles;
};

TfTransformStamped transform(const std::string& parentFrame,
                             const std::string& childFrame,
                             double x,
                             double y = 0.0,
                             double z = 0.0,
                             double timeStamp = 0.0)
{
    TfTransformStamped msg;
    msg.timeStamp = timeStamp;
    msg.parentFrame = parentFrame;
    msg.childFrame = childFrame;
    msg.translationX = x;
    msg.translationY = y;
    msg.translationZ = z;
    msg.rotationX = 0.0;
    msg.rotationY = 0.0;
    msg.rotationZ = 0.0;
    msg.rotationW = 1.0;
    return msg;
}

std::shared_ptr<FakeNodeHandle> makeNode(FakeContextHandle* context)
{
    return std::make_shared<FakeNodeHandle>(context);
}

} // namespace

TEST_SUITE("isaacsim.ros2.nodes.tf_aggregation_manager")
{
    // Verifies dynamic TF publishes the current post-OmniGraph epoch as one merged message.
    TEST_CASE("dynamic publishes current epoch")
    {
        TfAggregationManager manager;
        FakeFactory factory;
        FakeContextHandle context(12);
        Ros2QoSProfile qos;
        auto nodeA = makeNode(&context);
        auto nodeB = makeNode(&context);

        auto contributorA = manager.registerContributor("raw_a", &factory, nodeA, &context, "/tf", false, qos, false);
        auto contributorB = manager.registerContributor("raw_b", &factory, nodeB, &context, "/tf", false, qos, false);
        REQUIRE(contributorA);
        REQUIRE(contributorB);
        REQUIRE(factory.publishers.size() == 1);
        REQUIRE(factory.messages.size() == 1);

        CHECK(manager.submit(contributorA, 1.0, { transform("world", "base", 1.0, 0.0, 0.0, 1.0) }));
        CHECK(manager.submit(contributorB, 1.0, { transform("base", "camera", 0.0, 2.0, 0.0, 1.0) }));
        manager.flushAll();

        CHECK(factory.publishers[0]->publishCount() == 1);
        CHECK(factory.publishers[0]->lastMessage() == factory.messages[0]->getPtr());
        CHECK(factory.messages[0]->lastTimestamp == doctest::Approx(1.0));
        REQUIRE(factory.messages[0]->lastTransforms.size() == 2);
        CHECK(factory.messages[0]->lastTransforms[0].childFrame == "base");
        CHECK(factory.messages[0]->lastTransforms[0].translationX == doctest::Approx(1.0));
        CHECK(factory.messages[0]->lastTransforms[1].childFrame == "camera");
        CHECK(factory.messages[0]->lastTransforms[1].translationY == doctest::Approx(2.0));

        CHECK(manager.submit(contributorA, 2.0, { transform("world", "base", 2.0, 0.0, 0.0, 2.0) }));
        CHECK(manager.submit(contributorB, 2.0, { transform("base", "camera", 0.0, 3.0, 0.0, 2.0) }));
        manager.flushAll();

        CHECK(factory.publishers[0]->publishCount() == 2);
        CHECK(factory.messages[0]->lastTimestamp == doctest::Approx(2.0));
        REQUIRE(factory.messages[0]->lastTransforms.size() == 2);
        CHECK(factory.messages[0]->lastTransforms[0].translationX == doctest::Approx(2.0));
        CHECK(factory.messages[0]->lastTransforms[1].translationY == doctest::Approx(3.0));

        manager.flushAll();
        CHECK(factory.publishers[0]->publishCount() == 2);
    }

    // Verifies dynamic TF publishes changed transforms even when callers use a constant/default timestamp.
    TEST_CASE("dynamic publishes constant timestamp")
    {
        TfAggregationManager manager;
        FakeFactory factory;
        FakeContextHandle context(12);
        Ros2QoSProfile qos;
        auto nodeA = makeNode(&context);
        auto nodeB = makeNode(&context);

        auto contributorA = manager.registerContributor("raw_a", &factory, nodeA, &context, "/tf", false, qos, false);
        auto contributorB = manager.registerContributor("raw_b", &factory, nodeB, &context, "/tf", false, qos, false);
        REQUIRE(contributorA);
        REQUIRE(contributorB);

        CHECK(manager.submit(contributorA, 0.0, { transform("world", "base", 1.0) }));
        CHECK(manager.submit(contributorB, 0.0, { transform("base", "camera", 2.0) }));
        manager.flushAll();
        CHECK(factory.publishers[0]->publishCount() == 1);
        CHECK(factory.messages[0]->lastTimestamp == doctest::Approx(0.0));
        REQUIRE(factory.messages[0]->lastTransforms.size() == 2);
        CHECK(factory.messages[0]->lastTransforms[0].translationX == doctest::Approx(1.0));
        CHECK(factory.messages[0]->lastTransforms[1].translationX == doctest::Approx(2.0));

        CHECK(manager.submit(contributorA, 0.0, { transform("world", "base", 3.0) }));
        CHECK(manager.submit(contributorB, 0.0, { transform("base", "camera", 4.0) }));
        manager.flushAll();
        CHECK(factory.publishers[0]->publishCount() == 2);
        CHECK(factory.messages[0]->lastTimestamp == doctest::Approx(0.0));
        REQUIRE(factory.messages[0]->lastTransforms.size() == 2);
        CHECK(factory.messages[0]->lastTransforms[0].translationX == doctest::Approx(3.0));
        CHECK(factory.messages[0]->lastTransforms[1].translationX == doctest::Approx(4.0));
    }

    // Verifies a dynamic epoch can publish with only the contributors that submitted for it.
    TEST_CASE("dynamic missing contributor")
    {
        TfAggregationManager manager;
        FakeFactory factory;
        FakeContextHandle context(12);
        Ros2QoSProfile qos;
        auto nodeA = makeNode(&context);
        auto nodeB = makeNode(&context);

        auto contributorA = manager.registerContributor("raw_a", &factory, nodeA, &context, "/tf", false, qos, false);
        auto contributorB = manager.registerContributor("raw_b", &factory, nodeB, &context, "/tf", false, qos, false);
        REQUIRE(contributorA);
        REQUIRE(contributorB);

        CHECK(manager.submit(contributorA, 1.0, { transform("world", "base", 1.0, 0.0, 0.0, 1.0) }));
        manager.flushAll();
        CHECK(factory.publishers[0]->publishCount() == 1);
        CHECK(factory.messages[0]->lastTimestamp == doctest::Approx(1.0));
        REQUIRE(factory.messages[0]->lastTransforms.size() == 1);
        CHECK(factory.messages[0]->lastTransforms[0].childFrame == "base");

        CHECK(manager.submit(contributorA, 2.0, { transform("world", "base", 2.0, 0.0, 0.0, 2.0) }));
        CHECK(manager.submit(contributorB, 2.0, { transform("world", "camera", 3.0, 0.0, 0.0, 2.0) }));
        manager.flushAll();

        CHECK(factory.publishers[0]->publishCount() == 2);
        CHECK(factory.messages[0]->lastTimestamp == doctest::Approx(2.0));
        REQUIRE(factory.messages[0]->lastTransforms.size() == 2);
        CHECK(factory.messages[0]->lastTransforms[0].childFrame == "base");
        CHECK(factory.messages[0]->lastTransforms[1].childFrame == "camera");
    }

    // Verifies dynamic groups drop data without subscribers unless publish-without-verification is enabled.
    TEST_CASE("dynamic subscriber gate")
    {
        TfAggregationManager manager;
        FakeFactory factory;
        FakeContextHandle context(12);
        Ros2QoSProfile qos;
        factory.nextSubscriptionCount = 0;
        auto node = makeNode(&context);

        auto contributor = manager.registerContributor("raw", &factory, node, &context, "/tf", false, qos, false);
        REQUIRE(contributor);
        CHECK(manager.submit(contributor, 1.0, { transform("world", "base", 1.0, 0.0, 0.0, 1.0) }));
        CHECK(manager.submit(contributor, 2.0, { transform("world", "base", 2.0, 0.0, 0.0, 2.0) }));
        manager.flushAll();
        CHECK(factory.publishers[0]->publishCount() == 0);

        TfAggregationManager bypassManager;
        FakeFactory bypassFactory;
        bypassFactory.nextSubscriptionCount = 0;
        auto bypassNode = makeNode(&context);
        auto bypassContributor = bypassManager.registerContributor(
            "raw_bypass", &bypassFactory, bypassNode, &context, "/tf", false, qos, true);
        REQUIRE(bypassContributor);
        CHECK(bypassManager.submit(bypassContributor, 1.0, { transform("world", "base", 1.0, 0.0, 0.0, 1.0) }));
        CHECK(bypassManager.submit(bypassContributor, 2.0, { transform("world", "base", 2.0, 0.0, 0.0, 2.0) }));
        bypassManager.flushAll();
        CHECK(bypassFactory.publishers[0]->publishCount() == 1);
    }

    // Verifies static groups publish cached transforms once and republish only on changes or unregister.
    TEST_CASE("static dirty cache")
    {
        TfAggregationManager manager;
        FakeFactory factory;
        FakeContextHandle context(12);
        Ros2QoSProfile qos;
        auto nodeA = makeNode(&context);
        auto nodeB = makeNode(&context);

        auto contributorA =
            manager.registerContributor("static_a", &factory, nodeA, &context, "/tf_static", true, qos, false);
        auto contributorB =
            manager.registerContributor("static_b", &factory, nodeB, &context, "/tf_static", true, qos, false);
        REQUIRE(contributorA);
        REQUIRE(contributorB);

        CHECK(manager.submit(contributorA, 1.0, { transform("world", "base", 1.0, 0.0, 0.0, 1.0) }));
        CHECK(manager.submit(contributorB, 1.0, { transform("base", "camera", 0.0, 2.0, 0.0, 1.0) }));
        manager.flushAll();
        CHECK(factory.publishers[0]->publishCount() == 1);
        REQUIRE(factory.messages[0]->lastTransforms.size() == 2);

        CHECK(manager.submit(contributorA, 2.0, { transform("world", "base", 1.0, 0.0, 0.0, 2.0) }));
        CHECK(manager.submit(contributorB, 2.0, { transform("base", "camera", 0.0, 2.0, 0.0, 2.0) }));
        manager.flushAll();
        CHECK(factory.publishers[0]->publishCount() == 1);

        CHECK(manager.submit(contributorB, 3.0, { transform("base", "camera", 0.0, 3.0, 0.0, 3.0) }));
        manager.flushAll();
        CHECK(factory.publishers[0]->publishCount() == 2);
        REQUIRE(factory.messages[0]->lastTransforms.size() == 2);
        CHECK(factory.messages[0]->lastTransforms[1].translationY == doctest::Approx(3.0));

        manager.unregisterContributor(contributorB);
        manager.flushAll();
        CHECK(factory.publishers[0]->publishCount() == 3);
        REQUIRE(factory.messages[0]->lastTransforms.size() == 1);
        CHECK(factory.messages[0]->lastTransforms[0].childFrame == "base");
    }

    // Verifies unregister releases the departed contributor's node handle while the group remains active.
    TEST_CASE("unregister releases node handle")
    {
        TfAggregationManager manager;
        FakeFactory factory;
        FakeContextHandle context(12);
        Ros2QoSProfile qos;
        auto nodeA = makeNode(&context);
        auto nodeB = makeNode(&context);
        std::weak_ptr<FakeNodeHandle> nodeARef = nodeA;
        std::weak_ptr<FakeNodeHandle> nodeBRef = nodeB;

        auto contributorA = manager.registerContributor("raw_a", &factory, nodeA, &context, "/tf", false, qos, false);
        auto contributorB = manager.registerContributor("raw_b", &factory, nodeB, &context, "/tf", false, qos, false);
        REQUIRE(contributorA);
        REQUIRE(contributorB);

        nodeA.reset();
        nodeB.reset();
        CHECK_FALSE(nodeARef.expired());
        CHECK_FALSE(nodeBRef.expired());

        manager.unregisterContributor(contributorA);
        CHECK(nodeARef.expired());
        CHECK_FALSE(nodeBRef.expired());
        REQUIRE(factory.publisherNodeHandles.size() == 2);
        CHECK(factory.publisherNodeHandles[1] == nodeBRef.lock().get());
        CHECK(manager.submit(contributorB, 1.0, { transform("world", "base", 1.0, 0.0, 0.0, 1.0) }));

        manager.unregisterContributor(contributorB);
        CHECK(nodeBRef.expired());
    }

    // Verifies contributors share publishers only when topic, static flag, QoS, and ROS domain match.
    TEST_CASE("group key partitioning")
    {
        TfAggregationManager manager;
        FakeFactory factory;
        FakeContextHandle domainA(12);
        FakeContextHandle domainB(13);
        Ros2QoSProfile qos;
        Ros2QoSProfile deeperQos = qos;
        deeperQos.depth = qos.depth + 1;

        auto sameGroupA =
            manager.registerContributor("same_a", &factory, makeNode(&domainA), &domainA, "/tf", false, qos, false);
        auto sameGroupB =
            manager.registerContributor("same_b", &factory, makeNode(&domainA), &domainA, "/tf", false, qos, false);
        auto differentTopic =
            manager.registerContributor("topic", &factory, makeNode(&domainA), &domainA, "/tf_alt", false, qos, false);
        auto differentStatic =
            manager.registerContributor("static", &factory, makeNode(&domainA), &domainA, "/tf", true, qos, false);
        auto differentQos =
            manager.registerContributor("qos", &factory, makeNode(&domainA), &domainA, "/tf", false, deeperQos, false);
        auto differentDomain =
            manager.registerContributor("domain", &factory, makeNode(&domainB), &domainB, "/tf", false, qos, false);

        REQUIRE(sameGroupA);
        REQUIRE(sameGroupB);
        REQUIRE(differentTopic);
        REQUIRE(differentStatic);
        REQUIRE(differentQos);
        REQUIRE(differentDomain);
        CHECK(factory.publishers.size() == 5);
    }

    // Verifies duplicate child frames keep the first registered contributor transform.
    TEST_CASE("duplicate child frame")
    {
        TfAggregationManager manager;
        FakeFactory factory;
        FakeContextHandle context(12);
        Ros2QoSProfile qos;
        auto nodeA = makeNode(&context);
        auto nodeB = makeNode(&context);

        auto contributorA = manager.registerContributor("raw_a", &factory, nodeA, &context, "/tf", false, qos, false);
        auto contributorB = manager.registerContributor("raw_b", &factory, nodeB, &context, "/tf", false, qos, false);
        REQUIRE(contributorA);
        REQUIRE(contributorB);

        CHECK(manager.submit(contributorA, 1.0, { transform("world", "base", 1.0, 0.0, 0.0, 1.0) }));
        CHECK(manager.submit(contributorB, 1.0, { transform("odom", "base", 2.0, 0.0, 0.0, 1.0) }));
        manager.flushAll();

        CHECK(factory.publishers[0]->publishCount() == 1);
        REQUIRE(factory.messages[0]->lastTransforms.size() == 1);
        CHECK(factory.messages[0]->lastTransforms[0].parentFrame == "world");
        CHECK(factory.messages[0]->lastTransforms[0].translationX == doctest::Approx(1.0));
    }

    // Verifies invalid contributor handles are ignored and global manager access can be restored.
    TEST_CASE("invalid handle and global pointer")
    {
        TfAggregationManager manager;
        TfAggregationManager* originalManager = getTfAggregationManager();
        setTfAggregationManager(&manager);
        CHECK(getTfAggregationManager() == &manager);
        setTfAggregationManager(originalManager);
        CHECK(getTfAggregationManager() == originalManager);

        TfAggregationManager::ContributorHandle invalidHandle{ 999 };
        CHECK_FALSE(manager.submit(invalidHandle, 1.0, { transform("world", "base", 1.0) }));
        CHECK_NOTHROW(manager.unregisterContributor(invalidHandle));
        CHECK_NOTHROW(manager.unregisterContributor({}));
        CHECK_NOTHROW(manager.reset());
    }
}
